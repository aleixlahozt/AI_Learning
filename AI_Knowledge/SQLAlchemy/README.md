### Brief Tutorial: Getting Started with SQLAlchemy

**SQLAlchemy** is a popular Python SQL toolkit and Object-Relational Mapping (ORM) library that allows you to work with databases in a more Pythonic way. It supports both ORM (working with Python classes) and direct SQL execution.

Here's a step-by-step guide to getting started with SQLAlchemy, using both the **core** and **ORM** approaches.

---

### 1. **Installation**

First, install SQLAlchemy and a database driver (for example, **`psycopg2`** for PostgreSQL):

```bash
pip install sqlalchemy psycopg2
```

If you're using a different database (e.g., SQLite, MySQL, etc.), you'll need to install the appropriate driver.

---

### 2. **Setting Up the Database Connection**

Create an engine, which is the starting point for any SQLAlchemy application that communicates with a database.

```python
from sqlalchemy import create_engine

# Example for PostgreSQL
engine = create_engine("postgresql+psycopg2://user:password@localhost/mydatabase")

# For SQLite (no setup needed)
# engine = create_engine("sqlite:///mydatabase.db")

# For MySQL
# engine = create_engine("mysql+pymysql://user:password@localhost/mydatabase")
```

- Replace `user`, `password`, `localhost`, and `mydatabase` with your actual database credentials.

---

### 3. **Defining a Model (ORM)**

SQLAlchemy ORM allows you to define database tables as Python classes. Here's how you can define a model representing a `User` table:

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)

    def __repr__(self):
        return f"<User(name={self.name}, email={self.email})>"
```

- **`declarative_base()`**: This function returns a base class for defining mapped classes.
- **`__tablename__`**: Specifies the table name.
- **Columns**: Define table columns using `Column` and SQL types like `Integer` and `String`.

---

### 4. **Creating the Tables**

You need to create the tables in the database:

```python
Base.metadata.create_all(engine)
```

This creates the tables defined by your models in the connected database.

---

### 5. **Session Management**

A **session** is the way to interact with the database in SQLAlchemy ORM. It handles transactions and allows you to add, query, and delete data.

```python
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()
```

---

### 6. **CRUD Operations (Create, Read, Update, Delete)**

#### a. **Create a New Record**

```python
# Create a new user
new_user = User(name="John Doe", email="john@example.com")
session.add(new_user)
session.commit()  # Commit the transaction to save the changes
```

#### b. **Read Records**

```python
# Query the database for users
users = session.query(User).all()
for user in users:
    print(user)
```

#### c. **Update a Record**

```python
# Find the user and update the name
user = session.query(User).filter_by(name="John Doe").first()
user.name = "Jane Doe"
session.commit()
```

#### d. **Delete a Record**

```python
# Delete a user
user = session.query(User).filter_by(name="Jane Doe").first()
session.delete(user)
session.commit()
```

---

### 7. **Executing Raw SQL (Core)**

If you prefer writing raw SQL queries, SQLAlchemy Core allows you to execute SQL directly:

```python
from sqlalchemy import text

with engine.connect() as connection:
    result = connection.execute(text("SELECT * FROM users"))
    for row in result:
        print(row)
```

---

### 8. **Handling Transactions**

SQLAlchemy automatically wraps operations in a transaction, but you can explicitly handle them if needed:

```python
try:
    session.add(new_user)
    session.commit()  # Commit the transaction
except:
    session.rollback()  # Roll back in case of error
    raise
finally:
    session.close()  # Always close the session
```

---

### Summary:

- **SQLAlchemy Core**: For executing raw SQL queries and low-level database interaction.
- **SQLAlchemy ORM**: For working with databases using Python classes and objects.
- **Engine**: Represents the connection to the database.
- **Session**: Manages transactions and database operations.
- **Models**: Python classes that represent database tables.

This covers the basics of setting up and working with SQLAlchemy. You can extend this by learning more about relationships, joins, and advanced querying!