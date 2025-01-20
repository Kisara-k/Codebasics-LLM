import streamlit as st
from langchain_helper import generate

st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox("Pick a cuisine", ("Italian", "Chinese", "French", "Spanish", "Greek", "Indian", "Japanese", "Korean", "Mexican", "Thai"))


if cuisine:
    response = generate(cuisine)

    st.header(response['restaurant_name'].replace('"', ''))
    menu_items = response['menu_items'].split(',')
    st.write('**Menu Items:**')
    for item in menu_items:
        st.write(item.strip())