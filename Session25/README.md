# AI Math Agent with Paint Visualization 🧮🎨

## Overview
This project demonstrates an innovative approach to mathematical problem-solving using a Large Language Model (Gemini) combined with visual output through Microsoft Paint. The agent not only solves mathematical problems but also automatically visualizes the results.

<!-- ![System Architecture](docs/images/system_architecture.png)
*(Create an architecture diagram showing: LLM -> MCP Server -> Tools -> Paint)* -->

## System Architecture

```mermaid
graph LR
    User([User]) --> |Query| LLM[Gemini LLM]
    LLM --> |Tool Calls| MCP[MCP Server]
    
    subgraph Tools
        Math[Math Tools]
        String[String Tools]
        Paint[Paint Tools]
    end
    
    MCP --> |Executes| Tools
    Math --> |Results| MCP
    String --> |Results| MCP
    Paint --> |Visual Output| MSPaint[Microsoft Paint]
    
    MCP --> |Response| LLM
    LLM --> |Final Answer| User

    style LLM fill:#f9d,stroke:#fff
    style MCP fill:#9df,stroke:#fff
    style MSPaint fill:#dfd,stroke:#fff
```

## Features
- 🤖 Uses Gemini LLM for mathematical problem-solving
- 🛠️ Implements Multiple Tool Integration via MCP (Model-Control-Panel)
- 🎨 Automatic visualization in Microsoft Paint
- 📊 Real-time problem breakdown and solution steps
- 🔄 Iterative problem-solving approach

## System Architecture

### Components
1. **LLM (Gemini)**: Handles problem interpretation and solution strategy
2. **MCP Server**: Manages tool interactions
3. **Tool Suite**: 
   - Mathematical operations (add, subtract, multiply, etc.)
   - String manipulation
   - Paint operations (drawing, text addition)

<!-- ![Tool Interaction Flow](docs/images/tool_flow.png)
*(Create a flowchart showing how tools interact)* -->

### Tool Interaction Flow

```mermaid
flowchart TD
    Start([Query Start]) --> Parse[Parse Mathematical Query]
    Parse --> StringOp{Need String Operations?}
    
    StringOp -->|Yes| ASCII[Convert to ASCII]
    StringOp -->|No| MathOp{Need Math Operations?}
    
    ASCII --> MathOp
    
    MathOp -->|Yes| Calculate[Perform Calculations]
    MathOp -->|No| Visual{Need Visualization?}
    
    Calculate --> Visual
    
    Visual -->|Yes| Paint[Paint Operations]
    Visual -->|No| Result[Return Result]
    
    Paint --> OpenPaint[Open Paint]
    OpenPaint --> DrawRect[Draw Rectangle]
    DrawRect --> AddText[Add Text]
    AddText --> Result
    
    Result --> End([End])

    style Start fill:#f9f,stroke:#333
    style End fill:#f9f,stroke:#333
    style Paint fill:#9ef,stroke:#333
    style Calculate fill:#fe9,stroke:#333
```

## Example Use Case

```python
Query: "Find the ASCII values of characters in INDIA and then return sum of exponentials of those values"
```

### Solution Steps:
1. Convert "INDIA" to ASCII values
2. Calculate exponentials
3. Sum the results
4. Visualize in Paint

![Example Output](docs/images/example_output.png)
*(Add a screenshot of Paint output)*

## Installation

### Prerequisites
- Python 3.8+
- Microsoft Paint
- Gemini API access

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-math-agent.git

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your Gemini API key to .env
```

## Project Structure
```
.
├── Session25/
│   ├── talk2mcp.py      # Main application file
│   ├── example2.py      # MCP server implementation
├── requirements.txt
└── README.md
```

## How It Works

<!-- ![Sequence Diagram](docs/images/sequence_diagram.png)
*(Create a sequence diagram showing the interaction flow)* -->

### Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant L as LLM (Gemini)
    participant M as MCP Server
    participant T as Tools
    participant P as MS Paint

    U->>L: Submit Math Query
    activate L
    L->>M: Request Tool List
    M->>L: Return Available Tools
    
    L->>M: Call String Tool (ASCII conversion)
    M->>T: Execute String Operation
    T->>M: Return ASCII Values
    M->>L: Return Result
    
    L->>M: Call Math Tool (Exponential)
    M->>T: Calculate Exponentials
    T->>M: Return Calculations
    M->>L: Return Result
    
    L->>M: Call Paint Tool (open_paint)
    M->>P: Launch Paint
    P->>M: Paint Ready
    M->>L: Confirm Paint Open
    
    L->>M: Call Draw Rectangle
    M->>P: Draw Rectangle
    P->>M: Rectangle Drawn
    M->>L: Confirm Drawing
    
    L->>M: Call Add Text
    M->>P: Add Result Text
    P->>M: Text Added
    M->>L: Confirm Text
    
    L->>U: Return Final Answer
    deactivate L
```

1. **Problem Input**: User provides a mathematical query
2. **LLM Processing**: 
   - Analyzes the problem
   - Breaks it down into steps
   - Determines required tools
3. **Tool Execution**:
   - Makes appropriate tool calls
   - Processes intermediate results
4. **Visualization**:
   - Opens Paint
   - Creates visual frame
   - Displays result

## Tool List
- Mathematical Operations
  - `add(a: int, b: int) -> int`
  - `subtract(a: int, b: int) -> int`
  - `multiply(a: int, b: int) -> int`
  - More...
- String Operations
  - `strings_to_chars_to_int(string: str) -> list[int]`
- Paint Operations
  - `open_paint()`
  - `draw_rectangle(x: int, y: int, width: int, height: int)`
  - `add_text_in_paint(text: str)`

## Usage

```bash
python Session25/talk2mcp.py
```

## Demo
![Demo GIF](docs/images/demo.gif)
*(Create a GIF showing the entire process)*

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Gemini API for LLM capabilities
- MCP framework for tool integration
- Contributors and testers

## Future Improvements
- [ ] Add more mathematical operations
- [ ] Enhance visualization capabilities
- [ ] Support for complex mathematical notations
- [ ] Multiple visualization options
