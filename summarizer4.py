import os
from dotenv import load_dotenv

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv() # Loads the environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class SummarizerBase:
    """ Summarizer Class with multi-stage prompt definations """
    def __init__(self, texts, temperature, max_tokens, max_retries, use_gpt_model=True):
        self.texts = texts
        if use_gpt_model:
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            self.llm = OpenAI(
                temperature=temperature,
                model="text-davinci-003", #"gpt-3.5-turbo",
                max_tokens=max_tokens,
                max_retries=max_retries,
                frequency_penalty=0.5 # suppress the repeating token
            )

    def get_text_chunks(self, chunk_size=200, chunk_overlap=20):
        """ Split texts to handle long documents """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.create_documents([self.texts])

    def get_tokens_size(self):
        """ Get the tokens size """
        return self.llm.get_num_tokens(self.texts)

class SummarizerModel(SummarizerBase):
    """ Summarization using LLM Chain """
    def __init__(self, texts, temperature: float=0.2, max_tokens: int=1024, max_retries: int=2):
        super().__init__(texts, temperature, max_tokens, max_retries)

    def summarizer(self, min_tokens: int=20):
        """ Summarization using Chain of Density prompt """

        tokens_count = max(int(self.get_tokens_size()/3), min_tokens)

        map_template = """
            Article: {text}

            You will generate an increasingly concise, entity-dense summaries of the above Article given
            that you are a domain expert in humanitarian crisis analysis.

            Repeat the following 2 steps 5 times keeping in mind that you are humanitarian crisis expert.

            Step 1. Identify 1-3 informative Entities ("; " delimited) from the Article which are missing from the previously generated summary.
            Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.

            A Missing Entity is:
            - Relevant: to the main story.
            - Specific: descriptive yet concise (5 words or fewer).
            - Novel: not in the previous summary.
            - Faithful: present in the Article.
            - Anywhere: located anywhere in the Article.

            Guidelines:
            - Note to include the key figures and important events, dates, locations mentioned in the texts.
            - The first summary should be long ({tokens_count} words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
            - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
            - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
            - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
            - Missing entities can appear anywhere in the new summary.
            - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

            Remember, use the exact same number of words for each summary.

            Answer in Text. The Text should be the last generated summary.
        """

        template_prompt = PromptTemplate.from_template(map_template)

        summarize_chain = LLMChain(
            llm=self.llm,
            prompt=template_prompt
        )

        s = summarize_chain({"text": self.texts, "steps_freq": 5, "tokens_count": tokens_count})

        return s

    def get_summary(self, chunks, keywords, num_words):
        """ Get the Summary from the custom prompts """
        map_template = '''You are an expert in Humanitarian crisis document analysis. Gather all the important information including major events and the associated key persons or groups or organizations, locations, dates and numeric values from the following texts. It is important not to include information that is not present in the texts.
        Also give special emphasis to the sentences that contain these set of keywords {keywords}.

        Text:`{text}`

        Gathered Information:
        '''

        map_template_prompt = PromptTemplate(input_variables=['text', 'keywords'], template=map_template)

        combine_template = '''Generate a concise summary of the following texts including key points within {num_words} words. Make sure you do not add any false information and remove redundant information present in the texts.
        Note that sentences containing these list of keywords {keywords} get more emphasis.

        Text:`{text}`

        Concise Summary:
        '''

        combine_template_prompt = PromptTemplate(input_variables=['text', 'keywords', 'num_words'], template=combine_template)

        summarize_chain = load_summarize_chain(
            llm=self.llm,
            chain_type='map_reduce',
            map_prompt=map_template_prompt,
            combine_prompt=combine_template_prompt,
            input_key="input_documents",
            verbose=False
        )

        #return summarize_chain.run(chunks=chunks, keywords=keywords)
        return summarize_chain({"input_documents": chunks, "keywords": keywords, "num_words": num_words}, return_only_outputs=True)["output_text"]
