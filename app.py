import  streamlit as st
import pickle
import re
import nltk
nltk.download('punk')
nltk.download('stopwords')


# loading model

clf=pickle.load(open('clf.pkl', 'rb'))
tfidf= pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(resume_text):
    cleanText= re.sub('http:\S+\s','  ',resume_text)
    cleanText= re.sub('RT-cc','  ',cleanText)
    
    cleanText= re.sub('@\S+','  ',cleanText)
    cleanText= re.sub('#\S+\s','  ',cleanText)
    cleanText= re.sub('[%s]' % re.escape("""!"#$%&' ()*+,=./:;<=>?@[\]^_`{|}~"""),'  ',cleanText) #special character removal
    cleanText= re.sub(r'[^\x00-\x7f]','  ',cleanText)
    cleanText= re.sub('\s+','  ',cleanText)
    return cleanText


# webapp

def main():
    st.title("Resume Screener App")
    UploadFile= st.file_uploader('Upload Your Resume', type=['pdf','txt'])
    
    if UploadFile is not None:
        try:
            resume_bytes = UploadFile.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')  
                 # utf-18 and latin-1 are pdf files

        if isinstance(resume_text, str):
            CleanedResume = cleanResume(resume_text)  # Remove square brackets


        

        input_feature= tfidf.transform([CleanedResume])
        prediction_id= clf.predict(input_feature)[0]
        st.write(prediction_id)

        #Map Category ID to category name
        category_mapping={
            15:"Java developer",
            23:"Testing",
            8: "DevOps Engineer",
            20:"Python Developer",
            24:"Web Designing",
            12:"HR",
            13:"Hadoop",
            3:"BlockChain",
            10:"ETL Developer",
            18:"Operatios Manager",
            6:"Data Science",
            22:"Sales",
            16:"Mechanical Engineering",
            1:"Arts",
            7:"DataBase",
            11:"Electrical Engineering",
            14:"Health and fitness",
             4:"Business Analyst",
             9:"DotNet Developer",
            2:"Automation Testing",
            17:"Network Security Engineer",
            21:"SAP Developer",
            5:"Civil Engineer",
            0:"Advocate"

            }


        category_name= category_mapping.get(prediction_id, "unknown")
        st.write("Predicted Category Name:", category_name)

 #python main function

if __name__== "__main__" :
    main()  
