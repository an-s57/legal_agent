from rag.retriever import build_legal_vectorstore, add_legal_documents
import sys
if __name__=="__main__":
    if "--add" in sys.argv:
        print("添加新PDF到已有的向量库")
        add_legal_documents("legal_pdfs")
    else:
        print("全量重建向量库")
        build_legal_vectorstore("legal_pdfs")

print("建库完成")
#--add是自己增加的一个标记
