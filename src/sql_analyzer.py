import sqlite3
from sqlalchemy import create_engine, text
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import re

load_dotenv()

# Database
DB_PATH = "db/insurance.db"
DB_URL = f"sqlite:///{DB_PATH}"

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class SQLAnalyzerAgent:
    def __init__(self, db_url=DB_URL):
        # Connect to DB
        self.engine = create_engine(db_url)
        
        # Init Groq LLM
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile"  
        )
        
        # Prompt for NL → SQL
        self.prompt = PromptTemplate(
            input_variables=["question", "tables"],
            template="""
            You are an expert SQL assistant. Convert the user's question into a valid **SQLite SQL query**.

            - Use only the schema provided.
            - Do not add explanations or markdown fences.
            - Only return pure SQL.

            Schema:
            {tables}

            Question: {question}
            SQL:
            """
        )

    def get_schema(self):
        """Extract DB schema for grounding the LLM."""
        schema = []
        with self.engine.connect() as conn:
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' OR type='view'")).fetchall()
            for t in tables:
                table_name = t[0]
                cols = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                col_names = [c[1] for c in cols]
                schema.append(f"{table_name}({', '.join(col_names)})")
        return "\n".join(schema)

    def nl_to_sql(self, question):
        """Use Groq LLM to generate SQL from natural language."""
        schema = self.get_schema()

        # Use new pipeline style instead of deprecated LLMChain
        chain = self.prompt | self.llm
        sql = chain.invoke({"question": question, "tables": schema}).content.strip()

        # Remove markdown fences if LLM still adds them
        sql = re.sub(r"```sql|```", "", sql).strip()
        return sql

    def run_query(self, sql):
        """Run SQL query against DB."""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql)).fetchall()
        return result

    def ask(self, question):
        """End-to-end pipeline: NL → SQL → Result."""
        sql = self.nl_to_sql(question)
        print(f"\n📝 Generated SQL:\n{sql}\n")
        try:
            result = self.run_query(sql)
            return result
        except Exception as e:
            return f"❌ SQL Error: {e}"

# Example usage
if __name__ == "__main__":
    agent = SQLAnalyzerAgent()
    q = "Show me 5 clients with their sector and recommended products"
    result = agent.ask(q)
    print("📊 Query Result:\n", result)