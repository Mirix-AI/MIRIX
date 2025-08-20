import os
import json
import pandas as pd
from datetime import datetime
from datasets import load_dataset
import nltk
import tiktoken
from tqdm import tqdm
from constants import CHUNK_SIZE_MEMORY_AGENT_BENCH

class ConversationCreator():

    def __init__(self, dataset, num_exp, sub_datasets):
        
        self.dataset_name = dataset
        self.sub_datasets = sub_datasets
        self.num_exp = num_exp
        
        if dataset == "LOCOMO":
            with open("./data/locomo10.json", "r") as f:
                self.data = json.load(f)
        elif dataset == 'ScreenshotVQA':
            # Initialize data structure
            self.data = []
            with open("./data/ScreenshotVQA/qa_pairs.json", "r") as f:
                self.qa_pairs = json.load(f)

            # Base directory for ScreenshotVQA dataset
            base_dir = "./data/ScreenshotVQA"
            
            # List of student folders to process
            student_folders = ["student2", "student3", "student1"]

            # Process each student folder
            for student in student_folders:
                qa_pairs = [x for x in self.qa_pairs if x['student'] == student][0]['qa_pairs']
                student_dir = os.path.join(base_dir, student)
                if os.path.exists(student_dir):
                    # Get all image files in the student directory
                    image_files = [f for f in os.listdir(student_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    image_files.sort()
                    
                    # Create student entry with images and empty QA pairs
                    student_data = {
                        'images': [os.path.join(student_dir, img) for img in image_files],
                        'qas': qa_pairs
                    }
                    
                    self.data.append(student_data)
        elif dataset == 'MemoryAgentBench':
            # Load all 4 splits
            self.data = []
            # Login using e.g. `huggingface-cli login` to access this dataset
            ds = load_dataset("ai-hyz/MemoryAgentBench")
            splits = ['Accurate_Retrieval', 'Test_Time_Learning', 'Long_Range_Understanding', 'Conflict_Resolution']
            
            # For each sub_dataset, only select up to num_exp items.
            print(f"Loading {self.num_exp} items from each sub_dataset")
            sub_dataset_counts = {sub: 0 for sub in self.sub_datasets}
            for split in splits:
                df = ds[split]
                for row_dict in df:
                    source = row_dict['metadata']['source']
                    if source not in self.sub_datasets:
                        continue
                    # If num_exp is set, limit the number of items per sub_dataset
                    if self.num_exp is not None:
                        if sub_dataset_counts[source] >= self.num_exp:
                            continue
                        sub_dataset_counts[source] += 1
                    row_dict['split'] = split  # Append a key to record split the row belongs to
                    self.data.append(row_dict)
            # Context / Questions / Answers / Metadata
            print(f"Loaded {len(self.data)} items from MemoryAgentBench")
        else:
            raise NotImplementedError("Only LOCOMO and ScreenshotVQA datasets are supported")

    def chunk_text_into_sentences(self, text, model_name="gpt-4o-mini", chunk_size=4096):
        """
        Splits the input text into chunks of up to `chunk_size` tokens,
        making sure to split on sentence boundaries using NLTK's sent_tokenize.
        
        :param text: The long text document to be split.
        :param model_name: The name of the model to load the encoding for (default: gpt-3.5-turbo).
        :param chunk_size: Maximum number of tokens allowed per chunk.
        :return: A list of text chunks, each within the specified token limit.
        """
        # nltk.download('punkt_tab')
        
        # Initialize the tokenizer/encoding for the model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback if the model is not recognized by tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")

        # Break the text into sentences
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_chunk_token_count = 0

        for sentence in sentences:
            # Count tokens in this sentence
            sentence_tokens = encoding.encode(sentence, allowed_special={'<|endoftext|>'})
            sentence_token_count = len(sentence_tokens)
            
            # If adding this sentence exceeds the chunk_size, start a new chunk
            if current_chunk_token_count + sentence_token_count > chunk_size:
                # Push the current chunk as a single string
                chunks.append(" ".join(current_chunk))
                # Start a new chunk
                current_chunk = [sentence]
                current_chunk_token_count = sentence_token_count
            else:
                # Add this sentence to the current chunk
                current_chunk.append(sentence)
                current_chunk_token_count += sentence_token_count
        
        # Add the last chunk if there is any leftover
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def chunks(self, with_instructions=True):


        all_chunks = []
        
        if self.dataset_name == 'LOCOMO':
            for item in self.data:
                chunks = []
                conversation = item['conversation']
                for idx in range((len(conversation) - 2) // 2):
                    if with_instructions:
                        date_time = conversation['session_' + str(idx+1) + "_date_time"]
                        message = f"""You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

Instructions:

1. Carefully analyze all utterances from both speakers.
2. The conversation has a timestamp, but the events mentioned in the conversation may have different timestamps. You have to extract the exact date of the mentioned events. Remember that "mentioned at" is not the same as "occurred at" so this has to be noted in the memories.
3. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
4. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the conversation timestamp. 
5. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
6. You are supposed to extract the event/fact/semantic knowledge from the conversation. For example, if the conversation happens at 2023 and the conversation says that "John went to India last year", then you should save the fact that "John went to India in 2022". Similarly for all other kinds of memories.
7. Make sure to extract the facts about the characters, such as their name, age, gender, occupation, hometown, etc.

The conversation is shown below (the conversation is timestamped at {date_time}):\n\n"""
                    else:
                        date_time = conversation['session_' + str(idx+1) + "_date_time"]
                        message = f"Conversation happened at {date_time}:\n\n"
                    
                    try:
                        session = conversation['session_' + str(idx+1)]
                    except:
                        break
                    for turn in session:
                        message += turn['speaker'] + ": " + turn['text']
                        message += "\n"
                    chunks.append(message)
                all_chunks.append(chunks)
        
        elif self.dataset_name == 'ScreenshotVQA':

            all_chunks = []
            for item in self.data:
                chunks = []
                for image_idx, image in enumerate(item['images']):
                    if "compressed" in image:
                        continue
                    # Need to convert image "20250516_235610.png" into '%Y-%m-%d %H:%M:%S'
                    # Extract timestamp from filename (format: YYYYMMDD_HHMMSS.png)
                    filename = os.path.basename(image).split('.')[0]  # "20250516_235610"
                    date_part, time_part = filename.split('_')  # "20250516", "235610"
                    
                    # Parse date and time
                    year = date_part[:4]    # "2025"
                    month = date_part[4:6]  # "05"
                    day = date_part[6:8]    # "16"
                    hour = time_part[:2]    # "23"
                    minute = time_part[2:4] # "56"
                    second = time_part[4:6] # "10"
                    
                    # Format as datetime string
                    timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                    
                    chunks.append([(image, timestamp)])

                all_chunks.append(chunks)
                
        elif self.dataset_name == 'MemoryAgentBench':
            all_chunks = []
            for item in tqdm(self.data, desc=f"Processing {self.dataset_name} chunks", unit="item"):
                context = item['context']
                source = item['metadata']['source']
                chunks = []
                
                # Use a specific chunk size for each source
                chunk_size = CHUNK_SIZE_MEMORY_AGENT_BENCH[source]
        
                text_chunks = self.chunk_text_into_sentences(context, chunk_size=chunk_size)
                
                for chunk_text in text_chunks:
                    # Determine prompt based on source key
                    if source.startswith("longmemeval_s"):
                        message = f"Memorize the following conversation between the user and the assistant: \n {chunk_text} \n"
                    elif source == "recsys_redial_full":
                        message = f"Memorize the following dialogues between a user and recommender system: \n {chunk_text} \n"
                    elif source.startswith("factconsolidation"):
                        message = f"Memorize the following facts \n {chunk_text} \n"
                    else:
                        message = f"Memorize the following content: \n {chunk_text} \n"
                    
                    chunks.append(message)
                
                all_chunks.append(chunks)

        return all_chunks
    
    def get_query_and_answer(self):

        all_queries_and_answers = []

        if self.dataset_name == 'LOCOMO':
            for global_idx, item in enumerate(tqdm(self.data, desc=f"Processing {self.dataset_name} queries and answers", unit="item")):
                queries_and_answers = []
                for idx, qa in enumerate(item['qa']):
                    question = qa['question']
                    question = f"""You will be given a question and you need to answer the question based on the memories.
# APPROACH (Think step by step):
1. First, search and check the memories that might contain information related to the question.
2. Examine the timestamps and content of these memories carefully.
3. Look for explicit mentions of dates, times, locations, or events that answer the question.
4. If the answer requires calculation (e.g., converting relative time references), show your work.
5. Formulate a precise, concise answer based solely on the evidence in the memories.
6. Double-check that your answer directly addresses the question asked.
7. Ensure your final answer is specific and avoids vague time references like "yesterday", "last year" but with specific dates.
8. The answer should be as brief as possible, you should **only state the answer** WITHOUT repeating the question. For example, if asked 'When did Mary go to the store?', you should simply answer 'June 1st'. Do NOT say 'Mary went to the store on June 1st' which is redundant and strictly forbidden. Your answer should be as short as possible.

Question: {question}"""
                    try:
                        queries_and_answers.append(
                            [idx, question, qa['answer'], qa]
                        )
                    except:
                        continue
                
                all_queries_and_answers.append(queries_and_answers)

        elif self.dataset_name == 'ScreenshotVQA':
            all_queries_and_answers = []
            for item in tqdm(self.data, desc=f"Processing {self.dataset_name} queries and answers", unit="item"):
                queries_and_answers = []
                for idx, qa in enumerate(item['qas']):
                    question = qa['question']
                    answer = qa['answer']
                    queries_and_answers.append([idx, question, answer, qa])
                all_queries_and_answers.append(queries_and_answers)
                
        elif self.dataset_name == "MemoryAgentBench":
            from bench_template import get_template
            all_queries_and_answers = []
            for item in tqdm(self.data, desc=f"Processing {self.dataset_name} queries and answers", unit="item"):
                queries_and_answers = []
                
                if len(item['questions']) != len(item['answers']):
                    raise ValueError("Number of questions and answers are not the same")
                
                for idx in range(len(item['questions'])):
                    q_template = get_template(item['metadata']['source'], 'query', 'Long_context_agent')
                    
                    query = q_template.format(question=item['questions'][idx])
                    answer = item['answers'][idx]
                    # import pdb; pdb.set_trace()
                    queries_and_answers.append([idx, query, answer, item['metadata']['source']])
                all_queries_and_answers.append(queries_and_answers)
        
        return all_queries_and_answers

    def get_dataset_length(self):
        return len(self.data)