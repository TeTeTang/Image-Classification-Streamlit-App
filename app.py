import streamlit as st
from PIL import Image
from predict import predict
import os
import pandas as pd
import base64
from pandas import ExcelWriter


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def fn_gt(img):
    dispo_path, sys_path = "../../dispo", "../../system"
    if os.path.exists(dispo_path) or os.path.exists(sys_path):
    # if not os.path.exists(dispo_path) or not os.path.exists(sys_path):
        # return "No Ground Truth Label"
        dispo_set = set(os.listdir(dispo_path))
        system_set = set(os.listdir(sys_path))
        if img in dispo_set:
            return "dispo"
        elif img in system_set:
            return "system"
        # else:
        #     return "No Ground Truth Label"

def convert_df_xlsx(df):
    with ExcelWriter("Results.xlsx") as writer:
        # will return a None type
        return df.to_excel(writer)

# def to_excel(df):
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#     df.to_excel(writer, sheet_name='Sheet1')
#     writer.save()

def convert_df_csv(df):
    return df.to_csv().encode('utf-8')

@st.cache(allow_output_mutation=True)
def get_data():
    return []

def main():
    st.title("Image Classification App")
    st.write("")

    # By default, uploaded files are limited to 200MB. You can configure this using the server.maxUploadSize config option
    uploaded_files = st.file_uploader("Choose your images", accept_multiple_files=True, type=None)
    export_cols = {"PicName": [], "Label": [], "Probability": []}
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            st.image(load_image(uploaded_file))
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.write(file_details)
            # Prediction
            labels = predict(uploaded_file)
            gt = fn_gt(uploaded_file.name)
            export_cols["PicName"].append(uploaded_file.name)
            for i in labels:
                # st.write("Ground Truth Label: ", gt)
                # predict_words = '<p style="font-family:sans-serif; color:Green; font-size: 20px;">Predicted Class: </p>'
                # st.markdown(predict_words, unsafe_allow_html=True)
                st.write("Predicted Class: ", i[0])
                export_cols["Label"].append(i[0])
                # st.write("Prediction Probability: ", i[1] / 100)
                pred_prob = i[1] / 100
                export_cols["Probability"].append(pred_prob)
                # prob = st.slider('Confidence Score For The Prediction', 0.0, 1.0, pred_prob)
                if pred_prob < 0.75:
                    response = '<p style="font-family:sans-serif; color:Green; font-size: 16px;">How confident is the algorithm about the prediction: LOW</p>'
                    st.markdown(response, unsafe_allow_html=True)

                elif 0.75 <= pred_prob <= 0.8:
                    response = '<p style="font-family:sans-serif; color:Green; font-size: 16px;">How confident is the algorithm about the prediction: MEDIUM</p>'
                    st.markdown(response, unsafe_allow_html=True)
                    # st.write("The algorithm is *not very sure* about the prediction :frowning:")
                else:
                    response = '<p style="font-family:sans-serif; color:Green; font-size: 16px;">How confident is the algorithm about the prediction: HIGH</p>'
                    st.markdown(response, unsafe_allow_html=True)
                    # st.write("The algorithm is *quite sure* about the prediction :sunglasses:")

    primaryColor = st.get_option("theme.primaryColor")
    s = f"""
    <style>
    div.stButton > button:first-child {{ border: 5px solid {primaryColor}; border-radius:20px 20px 20px 20px; }}
    <style>
    """
    st.markdown(s, unsafe_allow_html=True)

# Export and Download the prediction file as a csv to your local machine
    if st.button('Export File!'):
        df_exp = pd.DataFrame(columns=list(export_cols.keys()))
        for pn, label, prob in zip(export_cols['PicName'], export_cols['Label'], export_cols['Probability']):
            to_append = [pn, label, prob]
            df_length = len(df_exp)
            df_exp.loc[df_length] = to_append
        st.write(df_exp)

        csv = convert_df_csv(df_exp)
        # excel = to_excel(df_exp)

        st.download_button(
            label="Download CSV File",
            data=csv,
            file_name='Results.csv',
            mime='text/csv',
            # file_name="Results.xlsx",
            # mime='text/xlsx'
        )

    # txt = st.text_area('Please leave your comments if you have questions about the ground truth and prediction results for some images', height=200)
    fn = st.text_area("If you have questions, please input the picture names here..., one per line.")
    comments = st.text_area('Please leave your comments..., one per line.', height=200)

    if st.button('Save Your Comments!'):
        fn_list = fn.split("\n")
        comment_list = comments.split("\n")
        for name, cm in zip(fn_list, comment_list):
            get_data().append({"picture name": name, "comments": cm})
        df = pd.DataFrame(get_data())
        st.write(df)
        open('comments record.csv', 'a').write(df.to_csv())


if __name__ == "__main__":
    main()
