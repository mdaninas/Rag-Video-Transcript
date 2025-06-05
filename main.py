from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableSequence

load_dotenv(find_dotenv())

path = "./transcription.txt"
loader = TextLoader(path)
document_temp = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
document_chunked = splitter.split_documents(document_temp)

vector_store = Chroma(
    collection_name="VecSt",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma"
)

prompt = ChatPromptTemplate.from_template(
    """
    Anda adalah asisten yang menjawab pertanyaan berdasarkan isi dokumen berikut:
    {context}

    Pertanyaan user:
    {question}

    Jika user bertanya \"apa isi video\", asumsikan bahwa user ingin mengetahui ringkasan dari dokumen tersebut.
    Jika tidak tahu, katakan \"Maaf, saya tidak tahu jawabannya\".
    """
)

llm = ChatOpenAI()

map_prompt = PromptTemplate.from_template(
    """
    Buatlah ringkasan singkat dan jelas dalam Bahasa Indonesia dari teks berikut:

    {context}
    """
)
map_chain = map_prompt | llm

combine_prompt = PromptTemplate.from_template(
    """
    Gabungkan ringkasan-ringkasan berikut menjadi ringkasan akhir dalam Bahasa Indonesia:

    {context}
    """
)
combine_chain = combine_prompt | llm

cached_summary = None
cached_speakers = None

speaker_prompt = PromptTemplate.from_template(
    """
    Berdasarkan teks berikut, siapa saja nama pembicara atau tokoh yang disebutkan?

    Teks:
    {context}
    """
)
speaker_chain = speaker_prompt | llm

def tanya_ke_video(pertanyaan):
    global cached_summary, cached_speakers

    if "apa isi video" in pertanyaan.lower():
        if cached_summary is None:
            summaries = [map_chain.invoke({"context": doc.page_content}).content for doc in document_chunked]
            cached_summary = combine_chain.invoke({"context": "\n\n".join(summaries)}).content
        print("\n\nRingkasan:\n", cached_summary)
        return

    elif "siapa" in pertanyaan.lower() and "speaker" in pertanyaan.lower():
        if cached_speakers is None:
            cached_speakers = speaker_chain.invoke({"context": document_temp[0].page_content}).content
        print("\n\nPembicara yang disebutkan:\n", cached_speakers)
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  
    docs = retriever.invoke(pertanyaan)

    if not docs:
        print("❗Tidak ditemukan konteks yang relevan untuk pertanyaan ini.")
        return

    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = prompt.format(context=context, question=pertanyaan)

    try:
        response = llm.invoke(formatted_prompt)
        print("\n\nJawaban:\n", response.content)
    except Exception as e:
        print("⚠️ Terjadi kesalahan saat memproses pertanyaan:", str(e))

pertanyaan = input("Apa yang ingin anda ketahui terhadap video tersebut? ")
tanya_ke_video(pertanyaan)
