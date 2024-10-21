import re
import torch
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from PyPDF2 import PdfReader
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import os
import PyPDF2
from groq import Groq
# Ensure that nltk data is downloaded
import nltk
import csv
nltk.download('punkt')

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self._extract_text_from_pdf()

    def _extract_text_from_pdf(self):
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join([page.extract_text() for page in reader.pages])
        return text

    def split_into_sentences(self):
        sentences = nltk.sent_tokenize(self._clean_text(self.text))
        return sentences

    def _clean_text(self, text):
        cleaned_text = ""
        for line in text.splitlines():
            cleaned_line = self._clean_string(line)
            if cleaned_line:
                cleaned_text += f"{cleaned_line} "
        return cleaned_text

    def _clean_string(self, s):
        s = re.sub(r'\bINFOID:\S*\b', '', s)
        single_letter_word = re.sub(r'[.\s]', '', s)
        return "" if len(single_letter_word) < 4 else s

class DataProcessor:
    def __init__(self):
        self.df = pd.DataFrame(columns=['Headings', 'Text', 'result'])
        self.df['result'] = 'Not Checked'
        self.heading = ''
        self.add_numeric = ''
        self.unique_sentences = []
        self.unique_embeddings = []
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')

    def populate_dataframe(self, sentences):
        for sentence in sentences:
            self._process_sentence(sentence)
        self._apply_heading_labels()
        self._print_heading_pivot()

    def _process_sentence(self, sentence):
        line_check = sentence[:10].split(':')[0]
        special_character_check = bool(re.match(r'^[\W_]+$', sentence[:3]))
        #print('step6')
        if bool(re.search(r'\d', ''.join(sentence.split()[:1]))):
            #print('step7')
            self.add_numeric = sentence
        elif line_check.isupper() and not special_character_check and not bool(re.search(r'\d', sentence[:1])):
            #print('step8')
            print(' '.join(re.findall(r'\b[A-Z]+(?=\b|:)', sentence)))
            heading = ' '.join(re.findall(r'\b[A-Z]+(?=\b|:)', sentence))
            #print('heading 1:',heading)
            #heading = ''.join([match + ':' if ':' in sentence[sentence.index(match) + len(match)] == ':' else match for match in heading])
            heading = ''.join([match + ':' if (sentence.index(match) + len(match) < len(sentence) and sentence[sentence.index(match) + len(match)] == ':') else match for match in heading])
            #print('heading 2:',heading)
            sentence = sentence.replace(heading, "")
            #print('step8A')
            if ''.join(sentence.split()) and all(char == '.' for char in ''.join(sentence.split())):
                #print('step8B')
                return
            else:
                #print('step8C')
                array = [heading.replace(":", ""), self.add_numeric + sentence, ""]
                #print('step8D')
                self.df.loc[len(self.df)] = array
                #print('step8E')
            self.add_numeric = ''
        elif not special_character_check and not bool(re.search(r'\d', ''.join(sentence.split()[:1]))):
            #print('step9')
            sentence = sentence.replace(self.heading, "")
            if ''.join(sentence.split()) and all(char == '.' for char in ''.join(sentence.split())):
                return
            else:
                array = [self.heading.replace(":", ""), self.add_numeric + sentence, ""]
                self.df.loc[len(self.df)] = array
            self.add_numeric = ''
        #print('step10')

    def _apply_heading_labels(self):
        # Ensure consistent label mapping
        label_mapping = {"CAUTION": "0", "DANGER": "1", "INSTRUCTION": "2", "PRECAUTION": "3", "WARNING": "4"}
        
        # Apply filtering sequentially
        self.df['Headings'] = self.df['Headings'].apply(lambda x: x if x in label_mapping else 'INSTRUCTION')
        self.df['Headings'] = self.df['Headings'].apply(lambda x: x if x in label_mapping else 'CAUTION')
        self.df['Headings'] = self.df['Headings'].apply(lambda x: x if x in label_mapping else 'PRECAUTION')

        # Mapping Headings to numeric values
        self.df['Headings'] = self.df['Headings'].map(label_mapping)
        
        
        # Debugging: Check the DataFrame after processing
        #print("DataFrame after applying heading labels:")
        #print(self.df[['Headings', 'Text']].head())

    def _print_heading_pivot(self):
        # Create a pivot table showing the count of rows for each heading
        pivot_df = self.df.pivot_table(index='Headings', values='Text', aggfunc='count')
        print("Pivot Table of Headings vs Count of Text Rows:")
        #print(pivot_df)

class FAISSIndexManager:
    def __init__(self, model_name,path, stage=None,df=None):
        self.model_name = model_name
        self.df = df
        self.csv_path = path+"/"+f'{self.model_name}_data.csv'
        self.vector_store_path = Path(path+"/"+f'pretrained_ST_{self.model_name}_vector_store.index')
        print('self.vector_store_path',self.vector_store_path)
        self.labels_path = Path(path+"/"+f'{self.model_name}_labels.npy')
        self.index = None
        self.labels = None
        
        if stage=='write':
            self.labels = self.df['Headings'].tolist()
            self._create_faiss_index()
        '''
        if self.df is None and self.csv_path:
            self.df = pd.read_csv(self.csv_path)
        
        
        else:
            raise ValueError("DataFrame or CSV path must be provided.")'''

    def _create_faiss_index(self):
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(self.df['Text'].tolist(), convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.vector_store_path))
        
        # Save labels
        np.save(self.labels_path, np.array(self.labels))
        
        # Save DataFrame to CSV
        self.df.to_csv(self.csv_path, index=False)

    def load_index(self, path):
        self.index = faiss.read_index(self.vector_store_path)
        if self.index is None:
            raise ValueError("Failed to load FAISS index. The index is None.")
        self.labels = np.load(self.labels_path).tolist()

        # Load DataFrame from CSV
        self.df = pd.read_csv(self.csv_path)
        if self.df.empty:
            raise ValueError("Failed to load DataFrame. The DataFrame is empty.")

    def retrieve_documents(self, query, k=50):
        self.index = faiss.read_index(str(self.vector_store_path))
        if self.index is None:
            raise ValueError("FAISS index is not initialized. Please load or create the index first.")

        model = SentenceTransformer(self.model_name)
        query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
        distances, indices = self.index.search(query_embedding, k + 20)
        print('distances, indices',len(distances[0]), len(indices[0]))
        return indices, distances




class RAGQueryManager:
    def __init__(self, model_name, df, index_manager,qamodelname,question):
        self.qa_pipeline = pipeline("question-answering", model=qamodelname)
        self.model_name = model_name
        self.df = df
        self.index_manager = index_manager
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.question=question
        self.reverse_label_mapping = {
            "0": "CAUTION",
            "1": "DANGER",
            "2": "INSTRUCTION",
            "3": "PRECAUTION",
            "4": "WARNING"
        }
        self.unique_sentences = []
        self.unique_embeddings = []

    def preprocess_text(self, text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def get_embedding(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            output = self.model(**tokens)
        # Mean pooling to get a single vector representation
        embeddings = output.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    def _is_unique_sentence(self, sentence):
        preprocessed_sentence = self.preprocess_text(sentence)
        new_embedding = self.get_embedding(preprocessed_sentence).reshape(1, -1)
        
        if not self.unique_sentences:  # If unique_sentences is empty
            self.unique_sentences.append(sentence)
            self.unique_embeddings.append(new_embedding)
            return True

        # Calculate cosine similarity with existing embeddings
        similarities = cosine_similarity(new_embedding, np.array(self.unique_embeddings).reshape(-1, new_embedding.shape[1]))
        if np.max(similarities) < 0.9:  # Check if the max similarity is below the threshold
            self.unique_sentences.append(sentence)
            self.unique_embeddings.append(new_embedding)
            return True

        return False
    def get_answer_or_context(self,sentence,question):
        result = self.qa_pipeline(question=self.question, context=sentence)
        answer = result['answer']
        sentence_embedding = self.get_embedding(sentence).reshape(1, -1)
        question_embedding = self.get_embedding(question).reshape(1, -1)
        answer_embedding = self.get_embedding(answer).reshape(1, -1)
        similarities = cosine_similarity(question_embedding, answer_embedding)[0][0]
        #print('print similratity',similarities,":",answer)
        #print(answer)
        #print('len(answer)',len(answer.split()))
        # Check if answer length is less than or equal to 3 characters
        if len(answer.split()) <= 3:
            similarities = cosine_similarity(question_embedding, sentence_embedding)[0][0]
            #print('print similratity',similarities,":",similarities)
            return sentence,similarities  # Return the entire sentence as the answer
        return answer,similarities  # Otherwise return the extracted answer

    def query(self, question, k=80):
        if len(self.df) <=80:
            k = len(self.df) - 20
        indices, cosine_similarities = self.index_manager.retrieve_documents(question, k)
        seen_documents = set()
        unique_results = []

        # Label count requirements for a total of 22 retrievals
        label_counts = {
            "2": 7,  # 10 instructions
            "1": 2,   # 3 danger
            "4": 2,   # 3 warning
            "0": 2,   # 3 caution
            "3": 2    # 3 precaution
        }
        collected_counts = {label: 0 for label in label_counts}
        # Filter out indices with negative values
        valid_indices = [index for index in indices[0] if index >= 0]

        # Ensure valid indices are within the range of the DataFrame
        valid_indices = [index for index in valid_indices if index < len(self.df)]
        #print('valid_indices',valid_indices)
        results_with_scores = list(zip(valid_indices, cosine_similarities[0]))
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
     

        answer_df = self.df.loc[valid_indices].reset_index()
        answer_df['Headings'] = answer_df['Headings'].apply(lambda label: self.reverse_label_mapping.get(str(label), str(label)))
    
        # Remove duplicate sentences
        unique_sentence_indices = {}
        
        for idx, sentence in enumerate(answer_df['Text']):
            if self._is_unique_sentence(sentence):
                unique_sentence_indices[sentence] = idx

        # Create a DataFrame with only the first occurrence of each unique sentence
        unique_indices = list(unique_sentence_indices.values())
        answer_df = answer_df.loc[unique_indices].reset_index(drop=True)
        #print('embedding_result',answer_df['Text'])

        # JM to export file for report -- embedding
        #answer_df.to_csv('embedding_layer.csv',index=False)

        answer_df[['Answer', 'cosine_sim']] = answer_df.apply(lambda row: pd.Series(self.get_answer_or_context(row['Text'], question)),axis=1)
        answer_df = answer_df[answer_df['cosine_sim'] > 0.6]
        answer_df = answer_df.sort_values(by='cosine_sim', ascending=False).reset_index(drop=True)
        # Get the current counts of each label in the DataFrame
        current_counts = answer_df['Headings'].value_counts()
        
        #print(current_counts)

        # Calculate shortfalls
        shortfalls = {label: max(0, label_counts.get(label, 0) - current_counts.get(self.reverse_label_mapping[label], 0)) for label in label_counts}
        print('shortfalls',shortfalls)
        # Filter DataFrame based on required counts
        filtered_dfs = []
        #print("test_filter",answer_df[answer_df['Headings'] == "CAUTION"].head(min(shortfalls["0"],label_counts.get("0", 0))))
        filtered_dfs.append(answer_df[answer_df['Headings'] == "CAUTION"].head(label_counts.get("0", 0)))
        filtered_dfs.append(answer_df[answer_df['Headings'] == "DANGER"].head(label_counts.get("1", 0)))
        filtered_dfs.append(answer_df[answer_df['Headings'] == "PRECAUTION"].head(label_counts.get("3", 0)))
        filtered_dfs.append(answer_df[answer_df['Headings'] == "WARNING"].head(label_counts.get("4", 0)))
        # Add instructions if needed (use key '2' for 'INSTRUCTION')
        total_instruction = max(label_counts["2"]+shortfalls["0"]+shortfalls["1"]+shortfalls["3"]+shortfalls["4"],label_counts["2"])

        #filtered_dfs.append(answer_df[answer_df['Headings'] == "INSTRUCTION"].head(total_instruction))
        filtered_dfs.append(answer_df[answer_df['Headings'] == "INSTRUCTION"].head(total_instruction))
        # Concatenate all filtered dataframes
        #filename = 'dilbert_output.csv'

        # Write the list to a CSV file
        answer_df = pd.concat(filtered_dfs).reset_index(drop=True)
        
        #print(answer_df)
        #answer_df['Answer'] = answer_df['Text'].apply(lambda sentence: self.get_answer_or_context(sentence, question))
        
        print("Current counts:", current_counts)
        print(f"Total results retrieved: {answer_df.shape}")

        return answer_df

def text_to_html(text):
    # Split text into sections based on double newlines
    sections = text.split('\n\n')

    # Create the base HTML structure
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        .header {
            font-weight: bold;
            font-size: 18px;
            margin-top: 20px;
        }
        p {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="header">Instructions for Working with High-Voltage Electrical Systems</div>
"""

    # Process each section
    for section in sections:
        section = section.strip()
        if section.startswith('**'):
            # It's a header
            header = section.strip('* ').strip()
            header = header.replace('*','').replace('\n', '<br>')
            html += f'    <div class="header">{header}</div>\n'
        elif section.startswith('*'):
            # It's a bullet point
            bullet_point = section.strip('* ').strip()
            bullet_point = bullet_point.replace('*','').replace('\n', '<br>')
            html += f'    <p>{bullet_point}</p>\n'
        elif re.match(r'^\d+\.\s', section):
            # It's a numbered list item
            # Split by numbers followed by a period and space
            items = re.split(r'(?=\d+\.\s)', section)
            for item in items:
                item = item.strip()
                if item:  # Avoid adding empty strings
                    item = item.replace('*','').replace('\n', '<br>')
                    html += f'    <p>{item}</p>\n'
        elif section:
            # It is a concluding paragraph or some other text
            section = section.replace('*','').replace('\n', '<br>')
            html += f'    <p>{section}</p>\n'

    # Close the body and html tags
    html += """</body>
</html>
"""
    return html


def get_chat_completion(prompt, model="llama3-70b-8192"):
    client = Groq(
        #api_key='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',
        api_key='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',
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                  #"content": prompt,
                "content": f"""
                {prompt}
                
                Instructions:
                - Summarize the key safety steps.
                - Highlight the most critical precautions.
                - Ensure the components are listed separately by new line under the section: Components Involved in the Process.
                - Ensure the response or instructions is structured in a clear, step-by-step manner.
                - give the output in html code format
                - Mark in red font for the identified and matching components in all the places.
                """,

            }
        ],
        model=model,
    )

    # Access the content directly from the message object
    return chat_completion.choices[0].message.content

def main(stage,faiss_model_names,qa_model_names,path,source_filename_pdf,question=None):
    result=pd.DataFrame()
    csv_path = path+f'{faiss_model_names}_data.csv'
    final_answers=""
    print(source_filename_pdf)
    if stage == 'write':
        print('source_filename_pdf',source_filename_pdf)
        pdf_path = source_filename_pdf
        #print('step1')
        pdf_processor = PDFProcessor(pdf_path)
        #print('step2')
        sentences = pdf_processor.split_into_sentences()
        #print('step3 sentences:',len(sentences))

        data_processor = DataProcessor()
        #print('step4')
        data_processor.populate_dataframe(sentences)
        #print('step5')
        #model_names = ['all-MiniLM-L6-v2']
        print('data_processor.df.shape',data_processor.df.shape)
        faiss_manager = FAISSIndexManager(faiss_model_names,path,stage, data_processor.df)

    if stage == 'query':

        if csv_path:
            df = pd.DataFrame(pd.read_csv(csv_path))

        print('df.shape',df.shape)
        faiss_manager = FAISSIndexManager(faiss_model_names,path,stage, df)
        rag_manager = RAGQueryManager(faiss_model_names, df, faiss_manager,qa_model_names,question )
        result = rag_manager.query(question)
        print('result.shape',result.shape)
        #print(result['Answer'])
        if result is None or result.empty:
            final_answers = "I dont understand your question. Can you please rephrase your question by adding components, are your trying to remove the component? etc."
        else:
            # JM to export file for report -- dilbert --
            #result.to_csv('dilbert_output.csv',index=False)
            response = ' '.join(result['Answer'].tolist()) +'.'
            text = get_chat_completion(response)
            #final_answers = text_to_html(text)
            final_answers = text


        #print(final_answers)
    return final_answers

'''
# Example usage
stage = 'query'
faiss_model_names = 'all-MiniLM-L6-v2'
qa_model_names = 'distilbert-base-uncased-distilled-squad'
#csv_path = f'{faiss_model_names}_data.csv'
path = f"test_program/"
result_df_final = main(stage,faiss_model_names,qa_model_names,path)'''