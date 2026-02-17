# Mermaid Syntax Reference

Comprehensive reference for Mermaid diagram syntax. Use this when writing `.mmd` files.

---

## Flowchart

```mermaid
flowchart TD
    A["Start"] --> B{"Decision?"}
    B -->|"Yes"| C["Action 1"]
    B -->|"No"| D["Action 2"]
    C --> E["End"]
    D --> E
```

### Direction
- `TD` or `TB` — Top to bottom
- `BT` — Bottom to top
- `LR` — Left to right
- `RL` — Right to left

### Node Shapes
```
A["Rectangle"]
B("Rounded rectangle")
C(["Stadium / pill"])
D[["Subroutine"]]
E[("Cylinder / database")]
F(("Circle"))
G{"Diamond / decision"}
H{{"Hexagon"}}
I>"Asymmetric / flag"]
J[/"Parallelogram"/]
K[\"Reverse parallelogram"\]
L[/"Trapezoid"\]
M[\"Reverse trapezoid"/]
N@{ shape: braces, label: "Curly braces" }
```

### Links / Edges
```
A --> B           solid arrow
A --- B           solid line (no arrow)
A -.-> B          dotted arrow
A -.- B           dotted line
A ==> B           thick arrow
A === B           thick line
A -->|"label"| B  labeled arrow
A -- "label" --> B  alternative label syntax
A <--> B          bidirectional
A ~~~ B           invisible link (for layout)
```

### Subgraphs
```mermaid
flowchart TD
    subgraph "Backend Services"
        direction LR
        API["API Server"]
        DB[("Database")]
        API --> DB
    end
    subgraph "Frontend"
        UI["Web App"]
    end
    UI --> API
```

### Styling
```mermaid
flowchart TD
    A["Success"]:::success --> B["Error"]:::error
    classDef success fill:#d4edda,stroke:#28a745,color:#155724
    classDef error fill:#f8d7da,stroke:#dc3545,color:#721c24
    classDef default fill:#e2e3e5,stroke:#6c757d,color:#383d41
    classDef highlight fill:#cce5ff,stroke:#004085,color:#004085
```

### Style individual links
```
linkStyle 0 stroke:red,stroke-width:2px
linkStyle default stroke:#333,stroke-width:1px
```

---

## Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant A as API
    participant DB as Database

    U->>+A: POST /login
    A->>+DB: Query user
    DB-->>-A: User record
    A-->>-U: 200 OK + token

    Note over A,DB: Auth flow complete

    alt Valid token
        U->>A: GET /data
        A-->>U: 200 OK
    else Invalid token
        U->>A: GET /data
        A-->>U: 401 Unauthorized
    end

    loop Every 30s
        U->>A: Heartbeat
    end

    rect rgb(200, 220, 255)
        Note right of U: Highlighted section
        U->>A: Special request
    end
```

### Arrow Types
```
->>    solid arrow (request)
-->>   dashed arrow (response)
-)     async message (open arrow)
--)    async dashed
->>+   activate target
-->>-  deactivate target
-x     solid cross (lost message)
--x    dashed cross
```

### Features
- `autonumber` — auto-number messages
- `participant X as "Label"` — alias participants
- `actor X as "Label"` — stick figure instead of box
- `Note over A,B: text` — spanning note
- `Note right of A: text` — positioned note
- `alt / else / end` — conditional
- `opt / end` — optional
- `loop / end` — loop
- `par / and / end` — parallel
- `critical / option / end` — critical section
- `break / end` — break out
- `rect rgb(r,g,b) / end` — highlight region
- `create participant X` — dynamic creation
- `destroy X` — dynamic destruction

---

## Class Diagram

```mermaid
classDiagram
    class Animal {
        +String name
        +int age
        +makeSound()* void
        +move() void
    }
    class Dog {
        +String breed
        +makeSound() void
        +fetch() void
    }
    class Cat {
        +makeSound() void
        +purr() void
    }

    Animal <|-- Dog : extends
    Animal <|-- Cat : extends
    Dog "1" --> "*" Toy : plays with
```

### Visibility
```
+ public
- private
# protected
~ package/internal
```

### Relationships
```
A <|-- B    Inheritance
A *-- B     Composition
A o-- B     Aggregation
A --> B     Association
A ..> B     Dependency
A ..|> B    Realization/Implementation
A -- B      Link (solid)
A .. B      Link (dashed)
```

### Cardinality
```
"1" -- "1"       one to one
"1" -- "*"       one to many
"1" -- "0..1"    one to zero or one
"*" -- "*"       many to many
```

### Annotations
```
<<interface>> ClassName
<<abstract>> ClassName
<<service>> ClassName
<<enumeration>> ClassName
```

---

## State Diagram

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing : submit
    Processing --> Success : valid
    Processing --> Error : invalid
    Error --> Idle : retry
    Success --> [*]

    state Processing {
        [*] --> Validating
        Validating --> Saving
        Saving --> [*]
    }

    state "Fork Example" as fork_state {
        state fork <<fork>>
        state join <<join>>
        [*] --> fork
        fork --> State1
        fork --> State2
        State1 --> join
        State2 --> join
        join --> [*]
    }

    note right of Idle : Waiting for input
```

### Features
- `[*]` — start/end pseudo-state
- `state "Label" as id` — aliased state
- Nested states with `state Parent { ... }`
- `<<fork>>`, `<<join>>` — concurrent states
- `<<choice>>` — choice pseudo-state
- `note right of State : text` — notes

---

## Entity-Relationship Diagram

```mermaid
erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE_ITEM : contains
    PRODUCT ||--o{ LINE_ITEM : "is in"
    CUSTOMER {
        int id PK
        string name
        string email UK
    }
    ORDER {
        int id PK
        int customer_id FK
        date created_at
        string status
    }
    LINE_ITEM {
        int id PK
        int order_id FK
        int product_id FK
        int quantity
    }
    PRODUCT {
        int id PK
        string name
        decimal price
    }
```

### Relationship Syntax
```
||--||    exactly one to exactly one
||--o{    one to zero or more
||--|{    one to one or more
o{--o{    zero or more to zero or more
```

### Cardinality symbols
```
||    exactly one
o|    zero or one
}|    one or more
}o    zero or more
```

### Attribute markers
- `PK` — primary key
- `FK` — foreign key
- `UK` — unique key

---

## Gantt Chart

```mermaid
gantt
    title Project Timeline
    dateFormat YYYY-MM-DD
    excludes weekends

    section Planning
        Requirements     :done, req, 2024-01-01, 7d
        Design           :done, des, after req, 5d

    section Development
        Backend API      :active, api, after des, 14d
        Frontend UI      :ui, after des, 14d
        Integration      :int, after api, 7d

    section Testing
        QA Testing       :qa, after int, 7d
        UAT              :uat, after qa, 5d

    section Launch
        Deployment       :milestone, deploy, after uat, 0d
```

### Task status
- `done` — completed
- `active` — in progress
- `crit` — critical path
- No status — future/pending
- `milestone` — milestone marker (0d duration)

---

## Pie Chart

```mermaid
pie title Distribution
    "Category A" : 40
    "Category B" : 30
    "Category C" : 20
    "Category D" : 10
```

---

## Mindmap

```mermaid
mindmap
    root((Central Topic))
        Branch 1
            Leaf 1a
            Leaf 1b
        Branch 2
            Leaf 2a
            Leaf 2b
                Sub-leaf
        Branch 3
            Leaf 3a
```

### Node shapes in mindmap
```
root((Circle))
    Default rectangle
    (Rounded)
    [Square]
    ))Bang((
    )Cloud(
    {{Hexagon}}
```

---

## Timeline

```mermaid
timeline
    title History of Events
    2020 : Event A
         : Event B
    2021 : Event C
    2022 : Event D
         : Event E
         : Event F
```

---

## Git Graph

```mermaid
gitgraph
    commit id: "Initial"
    branch develop
    checkout develop
    commit id: "Feature start"
    commit id: "Feature done"
    checkout main
    merge develop id: "Release v1"
    commit id: "Hotfix"
```

---

## User Journey

```mermaid
journey
    title User Shopping Journey
    section Browse
        Visit homepage: 5: User
        Search for item: 4: User
        View product: 4: User
    section Purchase
        Add to cart: 5: User
        Checkout: 3: User
        Payment: 2: User, System
    section Post-purchase
        Confirmation email: 5: System
        Delivery: 4: System
```

Scores are 1-5 (1 = negative, 5 = positive).

---

## Quadrant Chart

```mermaid
quadrantChart
    title Effort vs Impact
    x-axis Low Effort --> High Effort
    y-axis Low Impact --> High Impact
    quadrant-1 Do First
    quadrant-2 Schedule
    quadrant-3 Delegate
    quadrant-4 Eliminate
    Task A: [0.2, 0.8]
    Task B: [0.7, 0.9]
    Task C: [0.3, 0.3]
    Task D: [0.8, 0.2]
```

---

## Common Patterns & Tips

### Escaping special characters
- Wrap labels in double quotes when they contain special chars: `A["Node with (parens)"]`
- Use `#quot;` for quotes inside labels
- Use `#amp;` for ampersands
- Use `#lt;` and `#gt;` for angle brackets

### Layout control
- Use invisible links `A ~~~ B` to influence layout without visible connections
- Increase `width` and `height` in `mmdc` for complex diagrams
- Use `direction` inside subgraphs to mix LR and TD layouts
- Keep node IDs short (single letters or short abbreviations)

### Color palettes for classDef
```
Success:  fill:#d4edda,stroke:#28a745,color:#155724
Warning:  fill:#fff3cd,stroke:#ffc107,color:#856404
Error:    fill:#f8d7da,stroke:#dc3545,color:#721c24
Info:     fill:#cce5ff,stroke:#004085,color:#004085
Primary:  fill:#b8daff,stroke:#0056b3,color:#0056b3
Neutral:  fill:#e2e3e5,stroke:#6c757d,color:#383d41
Purple:   fill:#e8d5f5,stroke:#6f42c1,color:#4a0e78
```

### mmdc CLI usage
```bash
# Basic PNG generation
mmdc -i input.mmd -o output.png -q

# High-res with theme
mmdc -i input.mmd -o output.png -t forest -w 1200 -H 800 -s 2 -b white -q

# SVG output
mmdc -i input.mmd -o output.svg -t neutral -q

# PDF output
mmdc -i input.mmd -o output.pdf -f -q

# Dark theme with transparent background
mmdc -i input.mmd -o output.png -t dark -b transparent -s 2 -q
```

### Themes
- `default` — clean, professional, blue-toned
- `forest` — green-toned, nature palette
- `dark` — dark background, light text
- `neutral` — grayscale, minimal color
