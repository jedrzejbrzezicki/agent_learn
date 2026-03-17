from datetime import datetime
import pandas as pd
import re
from visualization_utils import  parse_plan_to_timeline


text= """
Here's your optimized daily plan:

- 14:00 - Lunch break (60min)

This is a short break to give you some time to refuel and recharge.

- 15:00 - LangChain Review (60min)

Start with reviewing LangChain material to reinforce your understanding, as this is a harder task.

- 16:00 - LangChain Study (30min)
- 16:30 - Break (10min)
- 16:40 - LangChain Study (30min)
- 17:10 - LangGraph (30min)
- 17:40 - Break (10min)
- 17:50 - LangGraph (30min)

This revised plan addresses your struggles by:

- Starting with LangChain Review, a harder task, to tackle it first
- Breaking down LangChain study into two manageable chunks
- Adding regular breaks to avoid burnout
- Prioritizing LangChain study in the morning and LangGraph in the evening
- Grouping similar activities (LangChain study and LangGraph) for flow state
"""




if __name__ == "__main__":
    df = parse_plan_to_timeline(text)
    assert len(df) == 8, f"Expected 8 activities, but got {len(df)}"

