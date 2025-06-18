import streamlit as st

def show_form(placeholder):
    st.header("Add new face")
    name = st.text_input("Enter name label:", key="name_input")
    if st.button("Add", key="add_button"):
        if name:
            st.session_state.page = "add_face_process"
            st.session_state.label_name = name
        else:
            st.error("Please enter a name!")
    placeholder.empty()  # Xóa placeholder để chuẩn bị cho quá trình add

