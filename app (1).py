import io, json, os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(page_title="Resistor Vision AI", page_icon="🔬", layout="wide")
st.title("🔬 Resistor Vision AI")
st.write("AI-powered resistor colour-band recognition and resistance calculation")
st.info("Use one resistor at a time, horizontally on a plain contrasting background. Avoid glare and keep all bands visible.")

DIGITS = {"Black":0,"Brown":1,"Red":2,"Orange":3,"Yellow":4,"Green":5,"Blue":6,"Violet":7,"Grey":8,"White":9}
MULTIPLIERS = {**{k:10**v for k,v in DIGITS.items()},"Gold":0.1,"Silver":0.01}
TOLERANCES = {"Brown":1,"Red":2,"Green":0.5,"Blue":0.25,"Violet":0.1,"Grey":0.05,"Gold":5,"Silver":10}
COLOURS = list(DIGITS) + ["Gold","Silver"]
MODEL_NAME = "gemini-3.6-flash"

def enhance_image(image):
    # Gentle luminance-only enhancement; chromatic channels are preserved.
    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8,8)).apply(l)
    soft = cv2.GaussianBlur(l, (0,0), 0.8)
    l = cv2.addWeighted(l, 1.18, soft, -0.18, 0)
    return Image.fromarray(cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB))

def image_part(image):
    image = image.convert("RGB").copy()
    image.thumbnail((1600,1600))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

def analyse_resistor(client, original, enhanced):
    prompt = f"""
Inspect the resistor in two images: original first, mildly luminance-enhanced copy second.
Use the original as authority for colour. Return ONLY JSON with keys:
is_resistor (boolean), band_count (0,4,5), colors_in_reading_order (array),
reading_order_confident (boolean), confidence (integer 0-100), notes (string).
Allowed colours: {", ".join(COLOURS)}.
Count only painted bands, not body colour, shadows, highlights, leads or background.
Return colours in electrical reading order: significant digits, multiplier, tolerance.
Do not guess. If count/order/colour is uncertain, set band_count=0 or
reading_order_confident=false and explain. If direction is ambiguous, do not choose.
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, image_part(original), image_part(enhanced)],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return json.loads(response.text)

def calculate_resistance(colors):
    if len(colors) == 4:
        x,y,m,t = colors
        if x not in DIGITS or y not in DIGITS or m not in MULTIPLIERS or t not in TOLERANCES:
            raise ValueError("Invalid 4-band colour sequence.")
        return (DIGITS[x]*10 + DIGITS[y])*MULTIPLIERS[m], TOLERANCES[t]
    if len(colors) == 5:
        x,y,z,m,t = colors
        if any(c not in DIGITS for c in (x,y,z)) or m not in MULTIPLIERS or t not in TOLERANCES:
            raise ValueError("Invalid 5-band colour sequence.")
        return (DIGITS[x]*100 + DIGITS[y]*10 + DIGITS[z])*MULTIPLIERS[m], TOLERANCES[t]
    raise ValueError("Expected exactly 4 or 5 bands.")

def fmt(v):
    if v >= 1e6: return f"{v/1e6:g} MΩ"
    if v >= 1e3: return f"{v/1e3:g} kΩ"
    return f"{v:g} Ω"

api_key = os.environ.get("GEMINI_API_KEY")
with st.sidebar:
    st.header("AI settings")
    st.caption(f"Model: {MODEL_NAME}")
    st.success("Gemini API key detected") if api_key else st.error("GEMINI_API_KEY not found in app environment.")

upload_tab, camera_tab = st.tabs(["📁 Upload image","📷 Camera capture"])
with upload_tab:
    upload = st.file_uploader("Upload resistor image", type=["jpg","jpeg","png"])
with camera_tab:
    st.caption("Camera mode captures a still frame. Continuous video analysis is not enabled in this version.")
    camera = st.camera_input("Capture resistor")
source = camera if camera is not None else upload

if source is not None:
    try:
        original = Image.open(source).convert("RGB")
        enhanced = enhance_image(original)
        c1,c2 = st.columns(2)
        with c1: st.image(original, caption="Original image", use_container_width=True)
        with c2: st.image(enhanced, caption="Enhanced image", use_container_width=True)

        if not api_key:
            st.error("The Streamlit process cannot see GEMINI_API_KEY. Launch it from Colab using the provided launch cell.")
        elif st.button("🔍 Analyse resistor", type="primary", use_container_width=True):
            try:
                client = genai.Client(api_key=api_key)
                with st.spinner("Gemini is inspecting the resistor bands..."):
                    result = analyse_resistor(client, original, enhanced)

                if not result.get("is_resistor"):
                    st.error("Gemini could not confirm a resistor.")
                    st.write(result.get("notes",""))
                elif result.get("band_count") not in (4,5):
                    st.warning("Gemini could not confidently identify 4 or 5 bands.")
                    st.write(result.get("notes",""))
                else:
                    colors = result.get("colors_in_reading_order", [])
                    confidence = max(0,min(100,int(result.get("confidence",0))))
                    valid = len(colors) == result["band_count"] and all(c in COLOURS for c in colors)
                    if not valid:
                        st.warning("The returned colour list is incomplete or invalid.")
                        st.write(result)
                    elif confidence < 70:
                        st.warning("Visual confidence is too low. Retake the image.")
                        st.write("Detected colours:", " – ".join(colors))
                    else:
                        # If Gemini is unsure about direction, test both possible
                        # reading directions against the standard 4-/5-band code.
                        # Use a direction only when exactly one is mathematically valid.
                        chosen_colors = colors
                        direction_note = ""
                        if not result.get("reading_order_confident", False):
                            candidates = []
                            for candidate in (colors, list(reversed(colors))):
                                try:
                                    candidate_value, candidate_tolerance = calculate_resistance(candidate)
                                    candidates.append((candidate, candidate_value, candidate_tolerance))
                                except ValueError:
                                    pass

                            if len(candidates) == 1:
                                chosen_colors, _, _ = candidates[0]
                                direction_note = "Reading direction resolved using the standard resistor colour-code rules."
                            elif len(candidates) == 2:
                                st.warning("Both directions form valid colour codes, so the reading direction is still uncertain. Please retake the photo with the tolerance band clearly separated.")
                                st.write("Detected colours:", " – ".join(colors))
                                st.stop()
                            else:
                                st.warning("The detected sequence does not form a valid standard resistor colour code.")
                                st.write("Detected colours:", " – ".join(colors))
                                st.stop()

                        try:
                            value,tolerance = calculate_resistance(chosen_colors)
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.success(f"Detected a {len(colors)}-band resistor")
                            st.subheader("Detected colour sequence")
                            swatches = {"Black":"#171717","Brown":"#8B4513","Red":"#DC2626","Orange":"#F97316","Yellow":"#EAB308","Green":"#16A34A","Blue":"#2563EB","Violet":"#8B5CF6","Grey":"#808080","White":"#F8FAFC","Gold":"#D4AF37","Silver":"#C0C0C0"}
                            cols = st.columns(len(colors))
                            for i,(col,colour) in enumerate(zip(cols,colors),1):
                                col.markdown(f'<div style="height:32px;border-radius:6px;background:{swatches[colour]};border:1px solid #888"></div>', unsafe_allow_html=True)
                                col.caption(f"Band {i}: {colour}")
                            st.subheader("Calculated resistance")
                            a,b,c = st.columns(3)
                            a.metric("Resistance",fmt(value))
                            b.metric("Tolerance",f"±{tolerance:g}%")
                            c.metric("Visual confidence",f"{confidence}%")
                            st.write(f"**Minimum:** {fmt(value*(1-tolerance/100))}")
                            st.write(f"**Maximum:** {fmt(value*(1+tolerance/100))}")
                            st.caption("Gemini identifies the colours; Python calculates resistance using the standard colour code. Verify before using in a circuit.")
                            if direction_note:
                                st.info(direction_note)
                            if result.get("notes"): st.write("**Inspection notes:**",result["notes"])
            except Exception as exc:
                st.error(f"Gemini analysis failed: {type(exc).__name__}: {exc}")
                st.info("Check model availability, API quota, internet connection and API key.")
    except Exception as exc:
        st.error(f"Could not read image: {exc}")

st.divider()
st.caption("Resistor Vision AI | Gemini vision + Python resistance decoder")
