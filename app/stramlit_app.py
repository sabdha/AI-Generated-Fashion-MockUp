import streamlit as st
from langgraph_flow.flow import build_flow
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
st.title("Fashion Product Mockup Generator")
style = st.text_input("Describe your product style (e.g. 'Red dress for beach shoot')")

if st.button("Generate Mockup"):
    with st.spinner("Generating your fashion mockup..."):
        flow = build_flow()
        results = flow.invoke({"query": style})
        for image_path in results["image_path"]:
            st.image(image_path, caption="AI-Generated Product Mockup")
        st.success("Done!")
