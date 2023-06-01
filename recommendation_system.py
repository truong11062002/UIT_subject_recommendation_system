import streamlit as st
import pandas as pd
from src.utils.add_condition import add_condition_description
from src.utils.get_items_prediction import get_items_prediction
from src.utils.input_prediction import input_prediction

def app():
    
    if st.sidebar.button('Refresh state'):
        st.session_state['add_data'] = False
        
    if 'add_data' not in st.session_state:
        st.session_state['add_data'] = False
        st.session_state['current_data'] = None
        
    # Load dataset
    dataset_original = pd.read_csv('./src/data/data_norm_final.csv')
    # -------------***********-------------
    
    # This is the sidebar
    st.sidebar.text('Choose an section')
    section = st.sidebar.selectbox('Pick one', ['Installation','Introduction', 'Recommendation System', 'Data Contribution'])
    # -------------***********-------------
    if section == 'Installation':
        pass
    
    # -------------***********-------------
    elif section == 'Introduction':
        st.image('./src/images/logo.jpg')
        # Title of your app
        st.markdown(
            '''
            ## Subject Recommendation System App
            ### Group 04: Final Project - Data mining and Applications
            #### Supervisor: Dr. Nguyen Thi Anh Thu
            #### Members:
            - Nguyễn Nhật Trường - 20522087
            - Huỳnh Viết Tuấn Kiệt - 20521494
            - Lại Chí Thiện - 20520309
            - Nguyễn Đức Anh Phúc - 20520276
            - Lê Thị Phương Vy - 20520355
            '''
        )     
    # Recommend based on the section chosen
    elif section == 'Recommendation System':
        dataset_original = dataset_original[[        \
            'mssv', 'gioitinh', 
            'CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT',   \
            'CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN',  \
            'mamh', 'tenmh', 'mota',    \
            'nganh_BB', 'nganh_BMAV', 'nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT',     \
            'diem_hp',      \
            'trangthai',        \
        ]]
        dataset = dataset_original.drop(dataset_original[(dataset_original['nganh_BB'] == 1) | (dataset_original['nganh_BMAV'] == 1)].index)
        dataset = dataset.drop_duplicates(subset=['mssv', 'mamh'], keep='last', inplace=False)
        dataset = add_condition_description(dataset)
        # -------------***********-------------
        # Create 3 datasets
        scores = dataset[['mssv', 'mamh', 'diem_hp']].copy().drop_duplicates()
        subjects = dataset[['mamh', 'tenmh', 'monhoc_encode', \
                            'nganh_BB', 'nganh_BMAV', 'nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT']].copy().drop_duplicates()
        students = dataset[['mssv', 'gioitinh', \
                            'CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT', \
                            'CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN']].copy().drop_duplicates()
        
        # -------------******TEST*****-------------
        # st.write(dataset.shape)
        # st.write(dataset_original.shape)
        # st.write(scores.shape)
        # st.write(subjects.shape)
        # st.write(students.shape)
        # st.dataframe(dataset.tail())
        
        attention_matrix = pd.read_csv('./src/data/attention_matrix.csv')
        # -------------******TEST*****-------------
        # st.write(attention_matrix.shape)
        # st.dataframe(attention_matrix)
        
        # -------------***********-------------
        # User input information student
        mssv = st.sidebar.text_input('Enter your id (MSSV)', placeholder='For example: 20522087, ...')
        
        options_rec = st.sidebar.radio('Choose type of recommendation', ['Recommend available information', 'Add information into system to recommend'])    
        
        if options_rec == 'Recommend available information':
            r_subject = st.sidebar.text_input('Enter subject you want to recommend', placeholder='For example: NT101, ...')
            st.write('You have chosen: ', r_subject)
            if st.sidebar.button('Recommend'):
                if r_subject in dataset_original["mamh"].unique():
                    st.write('Recommendation for you:')
                    try:
                        result_pred = get_items_prediction(
                            dataset=st.session_state['current_data'],
                            attn_matrix=attention_matrix, 
                            name_mh=r_subject,
                            mssv_query=mssv,
                            head_of_related_mh=20,
                            threshold_get_mssv=0.9
                        )
                    except:
                        st.write('You have not learned this subject yet')
                        result_pred = []
                    if len(result_pred) == 0:
                        st.write('No recommendation for you')
                    st.write(result_pred)
                else:
                    st.write('Subject not found in our program')
                    
        # -------------***********-------------            
        if options_rec == 'Add information into system to recommend':
            
            # -------------***********-------------
            sex = st.sidebar.radio('Enter your sex',['Nam', 'Nữ'])
            falculty = st.sidebar.selectbox('Enter your falculty',[
                'Công nghệ phần mềm', 
                'Hệ thống thông tin', 
                'Khoa học máy tính', 
                'Kỹ thuật máy tính',
                'Kỹ thuật thông tin',
                'Mạng máy tính và truyền thông'])
            type_of_education = st.sidebar.selectbox('Enter your type of education',[
                'Chất lượng cao',
                'Cử nhân tài năng',
                'Chính quy',
                'Chương trình tiên tiến',
                'Kỹ sư tài năng',
            ])
            # -------------***********-------------
            # Load list of subjects    
            list_of_subject = st.sidebar.multiselect(
            'Choose subjects you have passed',
            dataset_original['mamh'].unique())
            input_scores = [st.sidebar.number_input(f"Enter score for {subject}") for subject in list_of_subject]
            input_states = [st.sidebar.selectbox(f"Enter state for {subject}", ["Bình thường", "Nợ môn", "Cải thiện"]) for subject in list_of_subject]
            # -------------***********-------------
            
            if st.sidebar.button('Submit') and st.session_state['add_data'] == False:
                index = 0
                for subject, score in zip(list_of_subject, input_scores):
                    st.sidebar.write(f'{subject}: {score}')
                    dataset = dataset.append(input_prediction(
                        subjects=subjects,
                        _mssv = mssv,
                        _gioitinh = sex,
                        _khoa = falculty,
                        _hedaotao = type_of_education,
                        _mamh = subject,
                        # _ten_mh = 'Trí tuệ nhân tạo',
                        _diem_hp = score,
                        _trangthai = input_states[index] # 1: bình thường; 2: trả nợ; 3: cải thiện;
                    ), ignore_index=True)
                    index += 1
                st.session_state['add_data'] = True
                st.session_state['current_data'] = dataset.copy()
            
            # -------------***********-------------
            # st.write(st.session_state['add_data'])
            
            if st.session_state['add_data'] == True:
                
                r_subject = st.sidebar.text_input('Enter subject you want to recommend', placeholder='For example: NT101, ...')
                st.write('You have chosen: ', r_subject)
                if st.sidebar.button('Recommend'):
                    if r_subject in dataset_original["mamh"].unique():
                        st.write('Recommendation for you:')
                        try:
                            result_pred = get_items_prediction(
                                dataset=st.session_state['current_data'],
                                attn_matrix=attention_matrix, 
                                name_mh=r_subject,
                                mssv_query=mssv,
                                head_of_related_mh=20,
                                threshold_get_mssv=0.9
                            )
                        except:
                            st.write('You have not learned this subject yet')
                            result_pred = []
                        if len(result_pred) == 0:
                            st.write('No recommendation for you')
                        st.write(result_pred)
                    else:
                        st.write('Subject not found in our program')
        
# -------------***********-------------
# Run the Streamlit app
if __name__ == '__main__':
    app()