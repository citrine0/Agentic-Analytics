import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO
import sqlite3
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv

from langchain_community.chat_models import ChatZhipuAI
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_openai_tools_agent
from langchain.tools.render import render_text_description
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain.agents.output_parsers.react_json_single_input import ReActJsonSingleInputOutputParser
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import StrOutputParser
from langchain_core.agents import AgentFinish
from typing import Literal

# from langchain import hub
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents import create_sql_agent

load_dotenv()


# --- 0. 应用启动时的一次性设置 ---
def setup_app():
    output_dir = "dev_data"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "sales_data_200.csv")
    db_path = os.path.join(output_dir, "dev_database.db")
    if not os.path.exists(csv_path):
        st.toast("正在为您生成示例数据文件...")
        # ... (此处省略了数据生成的具体代码，与之前完全相同)
        PRODUCT_CATALOG = [
            ('智能手机 Pro', '电子产品', (4000, 8000)), ('蓝牙耳机 Air', '电子产品', (500, 1500)),
            ('笔记本电脑 Max', '电子产品', (6000, 12000)), ('机械键盘 K1', '电脑配件', (300, 800)),
            ('无线鼠标 M2', '电脑配件', (150, 400)), ('运动T恤', '服装', (80, 250)),
            ('休闲牛仔裤', '服装', (200, 600)), ('全自动咖啡机', '家居用品', (1000, 3000)),
            ('空气净化器', '家居用品', (800, 2500)),
        ]
        REGIONS = ['华东', '华北', '华南', '华中']
        PAYMENT_METHODS = ['支付宝', '微信支付', '信用卡', '花呗']

        def random_date(start, end):
            return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

        data = []
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        for i in range(1, 201):
            product_name, category, price_range = random.choice(PRODUCT_CATALOG)
            quantity = random.randint(1, 3)
            unit_price = round(random.uniform(price_range[0], price_range[1]), 2)
            row = {
                'order_id': 10000 + i, 'order_date': random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S"),
                'customer_id': random.randint(101, 150), 'product_name': product_name, 'category': category,
                'quantity': quantity, 'unit_price': unit_price, 'total_amount': round(quantity * unit_price, 2),
                'region': random.choice(REGIONS), 'payment_method': random.choice(PAYMENT_METHODS)
            }
            data.append(row)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        df.to_sql('sales', conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        st.toast("✅ 示例数据文件已生成！")


setup_app()

# --- 1. LLM 初始化 ---
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    st.error("❌ 错误：未找到 ZHIPUAI_API_KEY 环境变量。")
    st.info("请先设置您的智谱AI API Key: `export ZHIPUAI_API_KEY='your_key'`")
    st.stop()


@st.cache_resource
def get_llm():
    return ChatZhipuAI(model="glm-4-air-250414", temperature=0, api_key=api_key)


llm = get_llm()


# 为工具定义清晰的输入模型
class PythonCodeInput(BaseModel):
    code: str = Field(description="要执行的、用于数据分析或可视化的单行 Python 代码。")


@st.cache_resource
def create_agent_from_df(df: pd.DataFrame):
    """
    【最终电路测试版】
    放弃所有 Agent 和工具，只测试最核心的 LLM 代码生成能力。
    """
    # 1. 定义一个最简单的 Python REPL 环境
    repl = PythonAstREPLTool(locals={"df": df, "plt": plt, "pd": pd})

    # 2. 创建一个极其简单的 Prompt，只要求生成代码
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """
            你是一个 Python 代码生成器。根据用户问题，生成一段完整的、单行的、可以用分号分隔的 Python 代码，来对一个名为 `df` 的 DataFrame 进行操作。

            你的所有分析和代码【必须】只使用`df`中实际存在的列。在回答前，你必须在内心确认用户提到的列是否存在于`df`的结构中（参考下方提供的`df_schema`）。
            如果用户提到的关键列不存在，你【必须】生成一段返回Markdown字符串的Python代码，代码内容应明确告知用户数据缺失。在任何情况下，【绝对禁止】创造、猜测或假设任何不存在的数据。违反此规则将被视为重大失败。

            - 如果是数据查询，代码的最后一个表达式必须返回一个 Markdown 字符串。
            - 如果是绘图，代码应包含所有绘图指令。具体要求：
              - 代码开头必须包含 `plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]`。
              - 必须包含标题和坐标轴标签，如 `plt.title(...)`。
              - 为了美观，可以适当调整图形大小 `plt.figure(figsize=(10, 6))`。
              - 代码最后不需要有任何返回值。
            - 禁止包含任何 import 语句或重新定义 df。
            你的输出【必须】只有代码本身，不能有任何其他文字或解释。
             """),
            ("user",
             """
            # `df` 的结构:
            {df_schema}
            ---
            # 用户问题:
            {input}
            """),
        ]
    )

    # 3. 构建最简单的调用链: prompt -> llm -> string_output
    chain = prompt | llm | StrOutputParser()

    # 4. 创建一个简单的执行器类
    class FinalExecutor:
        def __init__(self, llm_chain, repl_tool):
            self.llm_chain = llm_chain
            self.repl_tool = repl_tool

        def invoke(self, input_dict: dict):
            try:
                # 第一步：调用 LLM 生成代码字符串
                print("--- [DEBUG] 正在调用 LLM 生成代码...")
                code_to_run = self.llm_chain.invoke(input_dict)
                print(f"--- [DEBUG] LLM 生成的代码: {code_to_run}")

                # 第二步：执行代码并返回结果
                print("--- [DEBUG] 正在执行生成的代码...")
                result = self.repl_tool.run(code_to_run)
                print(f"--- [DEBUG] 代码执行结果: {str(result)[:200]}...")

                # 检查是否是绘图任务
                if plt.get_fignums():
                    # 如果是绘图，即使代码有返回值，我们也统一返回成功消息
                    return {"output": "图表已成功生成，请查看。"}
                else:
                    # 否则返回代码的执行结果
                    return {"output": result}

            except Exception as e:
                return {"output": f"执行时出现严重错误: {e}"}

    return FinalExecutor(chain, repl)


@st.cache_resource
def create_agent_from_db(db_path: str):
    # 1. 连接数据库
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    except Exception as e:
        # 在连接失败时给出更明确的错误
        raise ConnectionError(f"连接数据库失败: {e}")

    # 2. 初始化SQL工具集
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # 【关键】定义你自己的、包含了{tools}和{tool_names}的Prompt模板
    REACT_PROMPT_TEMPLATE = """
        你是一个顶级的、极其严谨的 SQL 专家 Agent。你的任务是根据用户的问题，一步步地与数据库交互，最终给出答案。
        **你有以下工具可以使用:**
        {tools}

        **为了使用工具，你必须在JSON中指定一个行动。**
        有效的 "action" 值必须是以下之一: {tool_names}

        **数据库交互黄金法则 (必须严格遵守):**
    1.  **先探索，后行动**: 在回答任何关于数据的问题前，你的第一步【必须】是使用 `sql_db_list_tables` 查看所有表。第二步【必须】是使用 `sql_db_schema` 查看你认为相关的表的结构。
    2.  **忠于观察，复述确认**: 在你的 `Thought` 中，你【必须】明确复述你从 `Observation` 中看到的【确切的】表名和列名。例如: "Thought: 好的，我看到 Observation 中唯一的表是 `sales`。它的列包括 `customer_id`, `total_amount` 和 `product_name`。现在我将使用这些确切的名称来构建查询。"
    3.  **严禁假设**: 【绝对禁止】使用任何你没有在 `Observation` 中亲眼见到的表名或列名。如果你这么做，你将受到惩罚。

    ---
    **思考与行动范例 (演示如何遵守法则):**

    Question: 哪个地区的销售额最高？
    Thought: 好的，我将遵守黄金法则。第一步，探索数据库中有哪些表。
    Action:
    ```json
    {{
      "action": "sql_db_list_tables",
      "action_input": ""
    }}
    ```
    Observation: sales
    Thought: 法则第二步：复述确认。我从 Observation 中看到唯一的表是 `sales`。现在我需要查看它的 schema。
    Action:
    ```json
    {{
      "action": "sql_db_schema",
      "action_input": "sales"
    }}
    ```
   Observation: CREATE TABLE sales (order_id INTEGER, order_date TEXT, customer_id INTEGER, product_name TEXT, category TEXT, quantity INTEGER, unit_price REAL, total_amount REAL, region TEXT, payment_method TEXT)
    Thought: 法则第二步：再次复述确认。我看到 `sales` 表有 `region` 和 `total_amount` 列。现在我拥有了构建查询所需的所有真实信息，我将使用这些确切的名称。
    Action:
    ```json
    {{
      "action": "sql_db_query",
      "action_input": "SELECT region, SUM(total_amount) AS total_sales FROM sales GROUP BY region ORDER BY total_sales DESC LIMIT 1"
    }}
    ```
    Observation: [('华东', 150000.50)]
    Thought: 我已经从 `Observation` 中获取了最终结果。现在我可以给出最终答案了。
    Final Answer: 根据工具查询到的数据显示，销售额最高的地区是华东。 

    ---
     **最终答案生成指南 (必须遵守):**
    *   当工具的查询结果是一个列表（例如，包含多行数据）时，你的 `Final Answer` **必须** 将列表中的**每一项**都清晰、完整地列出来。
    *   **严禁**对多行结果进行任何形式的总结、截断或只选择其中一部分作为示例。你必须展示全部数据。
    现在，开始你的工作！严格遵守上述所有黄金法则。

    Question: {input}
    {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # 2.3 将 LLM 与停止序列绑定。这告诉 LLM 在生成 "Observation:" 后就停止，把控制权交回给 AgentExecutor。
    llm_with_stop = llm.bind(stop=["\nObservation:"])

    # 2.4 定义一个函数来格式化中间步骤 (agent action, observation)
    def format_log_to_str(intermediate_steps):
        """将中间步骤的日志转换为agent_scratchpad期望的字符串格式。"""
        log = ""
        for action, observation in intermediate_steps:
            # action.log 包含了 Thought 和 Action 的 JSON 块
            log += action.log
            log += f"\nObservation: {str(observation)}\n"
        return log

        # 2.5 【最关键的一步】手动构建 Agent 链
        # 这条链清晰地定义了数据流：
        # 1. 接收输入 (input, intermediate_steps)
        # 2. 使用 format_log_to_str 函数处理 intermediate_steps，生成 agent_scratchpad
        # 3. 将所有变量填充到 prompt 中
        # 4. 调用 llm_with_stop
        # 5. 使用正确的 ReActJsonSingleInputOutputParser 来解析 LLM 的输出

    agent: Runnable = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | prompt
            | llm_with_stop
            | ReActJsonSingleInputOutputParser()
    )

    # # 3. 创建为 ReAct 模式设计的 Agent
    # # 这个 Agent 被设计为自主探索数据库，所以我们不需要预先提供 Schema
    # agent = create_react_agent(
    #     llm=llm,
    #     tools=tools,
    #     prompt=prompt  # 使用我们为SQL ReAct精心设计的Prompt
    # )

    # 4. 创建 Agent Executor，并增加迭代次数限制
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        # 提供一个有用的错误提示，以防万一
        handle_parsing_errors=True,  # "请检查你的JSON格式，确保`action`的值是有效的工具名称或 'Final Answer'。",
        # 为复杂的、需要多步查询的任务设置一个更长的最大迭代次数
        max_iterations=15
    )

    return agent_executor


# --- 3. Streamlit 页面布局 (最终正确版) ---
# --- 初始化 Session State ---
# 确保 messages 列表在 session state 中只被初始化一次
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# --- 侧边栏逻辑 ---
with st.sidebar:
    st.header("数据类型")

    # 为清除按钮添加唯一的 key，防止重复 ID 错误
    if st.button("清除/重置数据和对话", key="clear_session_button"):
        # 使用 .clear() 来安全地重置所有 session 状态
        st.session_state.clear()
        st.rerun()

    # 为 radio 组件添加唯一的 key
    data_source_option = st.radio(
        "选择你的数据源:",
        ('上传CSV文件', '上传SQLite数据库', '使用内置示例数据'),
        index=2,
        key="data_source_radio"
    )

    # --- 数据加载逻辑 ---
    # 这部分保持您原来的逻辑即可
    if data_source_option == '上传CSV文件':
        uploaded_file = st.file_uploader("拖拽或点击上传CSV", type="csv", key="csv_uploader")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # 将关键对象存入 session state
                st.session_state.agent_executor = create_agent_from_df(df)
                st.session_state.dataframe = df
                st.session_state.agent_type = "dataframe"
                st.success("✅ CSV已加载，可以开始提问了！")
                with st.expander("数据预览 (前5行)"):
                    st.dataframe(df.head())
            except Exception as e:
                st.error(f"加载CSV文件失败: {e}")

    elif data_source_option == '上传SQLite数据库':
        uploaded_file = st.file_uploader("拖拽或点击上传DB文件", type=["db", "sqlite", "sqlite3"], key="db_uploader")
        if uploaded_file:
            temp_db_path = f"./temp_{uploaded_file.name}"
            with open(temp_db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                st.session_state.agent_executor = create_agent_from_db(temp_db_path)
                st.session_state.agent_type = "database"
                st.success("✅ DB已连接，可以开始提问了！")
            except Exception as e:
                st.error(f"连接数据库失败: {e}")

    elif data_source_option == '使用内置示例数据':
        st.info("我们将使用自动生成的`sales_data_200.csv`文件。")
        sample_csv_path = "dev_data/sales_data_200.csv"
        try:
            df = pd.read_csv(sample_csv_path)
            st.session_state.agent_executor = create_agent_from_df(df)
            st.session_state.dataframe = df
            st.session_state.agent_type = "dataframe"
            st.success("✅ 示例数据已加载，可以开始提问了！")
            with st.expander("示例数据预览 (前5行)"):
                st.dataframe(df.head())
            with open(sample_csv_path, "rb") as f:
                st.download_button("下载示例CSV文件", f, "sales_data_200.csv", "text/csv", key="download_csv_button")
        except FileNotFoundError:
            st.error("示例文件不存在，请重启应用以自动生成。")
        except Exception as e:
            st.error(f"加载示例文件失败: {e}")

# 定义示例问题和回调函数
EXAMPLE_PROMPTS = {
    "dataframe": [
        "请用条形图展示每个产品类别的总销售额。要求：图表标题为‘各产品类别销售额对比’，Y轴为‘总销售额’，并将条形图颜色设为专业的天蓝色（SteelBlue）。",
        "请用带有数据标记的折线图分析月度销售总额的趋势。要求：图表标题为‘月度销售总额趋势（2024年）’，X轴为‘月份’，Y轴为‘总销售额’，并将线条颜色设为经典的深蓝色。",
        "请用饼图展示不同支付方式的使用占比。要求：图表标题为‘支付方式分布情况’，并使用explode效果突出显示占比最高的那一项。",
        "分析不同地区的平均订单金额，并用水平条形图进行可视化。要求：图表标题为‘各地区平均订单金额对比’，X轴标签为‘平均金额（元）’，并将图表颜色设为清爽的绿色。",
    ],
    "database": [
        "查询并返回总销售额最高的地区。请用Markdown的格式清晰地展示地区名称和其对应的总销售额。",
        "列出销量最高的前5个产品是什么？请用一个带表头的Markdown表格来展示，包含‘产品名称’和‘总销量’两列。",
        "查询 '华南' 地区的所有订单记录，并以Markdown表格的形式返回前5条记录，展示订单ID、产品名称和总金额。",
        "统计每个支付方式被使用了多少次。请用一个按使用次数降序排列的Markdown表格显示结果，包含‘支付方式’和‘使用次数’两列。",
    ]
}


def on_example_click(prompt_text):
    """点击示例问题按钮时的回调函数"""
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.session_state.processing = True


# 使用列布局来放置标题和按钮
col_title, col_button = st.columns([0.7, 0.3])  # 分配宽度比例

with col_title:
    st.title("你的数据分析助手")

with col_button:
    # 仅在Agent就绪时才显示这个popover按钮
    if st.session_state.get("agent_executor"):
        st.write("")
        st.write("")
        with st.popover("➕ 示例问题"):
            st.markdown("您可以试试这些问题：")
            agent_type = st.session_state.get("agent_type", "dataframe")
            prompts_to_show = EXAMPLE_PROMPTS.get(agent_type, [])
            for prompt_example in prompts_to_show:
                st.button(
                    prompt_example,
                    on_click=on_example_click,
                    args=(prompt_example,),
                    use_container_width=True,
                    key=f"example_{prompt_example}"
                )

# --- 主聊天界面 ---

# 第一步：无条件地、始终在页面顶部渲染所有历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], plt.Figure):
            st.pyplot(message["content"])
        else:
            st.markdown(message["content"])

# 第二步：使用一个“状态锁”来防止重复调用 Agent
# 如果 session_state 中没有 'processing' 这个键，就初始化为 False
if 'processing' not in st.session_state:
    st.session_state.processing = False

# 第二步：处理用户的新输入。这个 if 块只负责“处理逻辑”和“更新状态”
if prompt := st.chat_input("请输入您感兴趣的分析问题..."):
    # a. 检查 Agent 是否已准备好
    if "agent_executor" not in st.session_state or st.session_state.agent_executor is None:
        st.warning("⚠️ 请先在侧边栏选择并加载您的数据源。")
        st.stop()

    # b. 将用户的【新消息】添加到 session_state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.processing = True
    st.rerun()

if st.session_state.processing:
    # a. 获取最后一条用户消息
    user_message = st.session_state.messages[-1]["content"]
    # c. 调用 Agent 并获取响应

    try:
        # 这个 with 块只用来显示“思考中”的动画
        with st.spinner("助手正在分析中... ⚙️"):
            plt.close('all')  # 清理任何可能存在的旧图表
            agent_type = st.session_state.get("agent_type")
            response = None

            if agent_type == "dataframe":
                df = st.session_state.dataframe
                buffer = StringIO()
                df.info(buf=buffer)
                df_schema = buffer.getvalue()
                input_dict = {"input": user_message, "df_schema": df_schema}
                response = st.session_state.agent_executor.invoke(input_dict)
            elif agent_type == "database":
                # 注意：你的 SQL Agent Prompt 没有 {chat_history} 变量，所以不传递它
                input_dict = {"input": user_message, "chat_history": []}
                response = st.session_state.agent_executor.invoke(input_dict)

            # d. 将助手的【文本回复】添加到 session_state
            # 增加一个 strip() 来清理可能存在的前后空格
            assistant_response_content = response.get("output",
                                                      "分析完成，但没有文本输出。").strip() if response else "未能从Agent获取响应。"
            st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})

            # e. 检查是否有图表，如果有，将【图表对象】也作为一条独立消息添加到 session_state
            if plt.get_fignums():
                fig = plt.gcf()
                st.session_state.messages.append({"role": "assistant", "content": fig})
                # 注意：不要在这里 close(fig)，因为 Streamlit 可能需要它来重新渲染
                # plt.close('all') 在下次调用前清理即可

    except Exception as e:
        error_message = f"分析时出现错误: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})

    # c.【关键】处理完毕后，关闭“状态锁”
    st.session_state.processing = False
    # d.【关键】触发最后一次 rerun，以显示 Agent 的完整回复
    st.rerun()





