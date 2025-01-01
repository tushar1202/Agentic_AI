from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import phi.api
from dotenv import load_dotenv
from phi.model.openai import OpenAIChat
import os
import phi
from phi.playground import Playground, serve_playground_app

# Load the .env file
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

## Web Search Agent
web_search_agent = Agent(
    name='Web Search Agent',
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information"],
    show_tool_calls=True,
    markdown=True 
)

## Financial Agent
finance_agent = Agent(
    name='Finance AI Agent',
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True 
)

app=Playground(agents=[web_search_agent, finance_agent]).get_app()

if __name__ == '__main__':
    serve_playground_app("playground:app",reload=True, port=8888)