# Fast API: [tutorial](https://fastapi.tiangolo.com/tutorial/first-steps/)

* High-performance web framework for building APIs with Python
* Extremely fast, faster than Django/Flask
  * Flask runs on WSGI (not asynchronously)
* Built-in Data Validation with Pydantic. Uses type hints to automatically validate data. Allows for custom types
* Errors are in JSON
* Built-in Authentication. Supports HTTP Basic OAuth2 tokens (JWT tokens) and Header API Keys
* Automatically generates interactive API documentation using SwaggerUI and Redoc
  * /doc
  * /redoc
* Automatically generates a JSON (schema) with the descriptions of all your API
  * /openapi.json

### To run:

1. pip install "fastapi[standard]"
2. fastapi dev main.py

## Comparison vs Flask, Django

Here’s a detailed markdown table comparing **Flask**, **Django**, and **FastAPI** with their main features, pros, cons, and recommendations for when to use each.

| **Feature**                 | **Flask**                                      | **Django**                                   | **FastAPI**                                |
| --------------------------------- | ---------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------ |
| **Type**                    | Microframework                                       | Full-stack Web Framework                           | Modern API framework (async-first)               |
| **Language**                | Python                                               | Python                                             | Python                                           |
| **Asynchronous Support**    | Partial (via async views or extensions)              | Partial (added in Django 3.1 for async views)      | Full support (built on ASGI with async)          |
| **Performance**             | Moderate (depends on extensions)                     | Moderate (due to ORM and other overhead)           | High performance (due to async and Uvicorn)      |
| **Routing**                 | Simple, minimal routing                              | Built-in, organized around views and URL configs   | Fast routing with data validation via Pydantic   |
| **Templating**              | Jinja2 (included)                                    | Django Templating Engine (included)                | Optional (Jinja2 or custom, if needed)           |
| **Database Support**        | No built-in ORM (use SQLAlchemy or others)           | Built-in ORM                                       | No built-in ORM (use SQLAlchemy or Tortoise ORM) |
| **Form Handling**           | Via extensions (e.g., Flask-WTF)                     | Built-in form handling and validation              | JSON and Pydantic-based validation               |
| **Authentication**          | Via extensions (e.g., Flask-Login)                   | Built-in authentication system                     | Via third-party libraries (e.g., FastAPI Users)  |
| **Admin Interface**         | None (extensions like Flask-Admin)                   | Built-in Admin Panel                               | None (use libraries or custom implementations)   |
| **REST API Support**        | Requires Flask-RESTful or similar extensions         | Built-in with Django Rest Framework (DRF)          | Built-in, automatically generated API docs       |
| **WebSockets Support**      | Via extensions (Flask-SocketIO)                      | Via third-party packages                           | Built-in (ASGI native, WebSockets support)       |
| **Auto-Generated API Docs** | No                                                   | No                                                 | Yes (Swagger UI and ReDoc)                       |
| **Dependency Injection**    | No (handled manually or with extensions)             | No (handled manually or with third-party packages) | Yes (built-in support via FastAPI's design)      |
| **Validation**              | No built-in (handled via libraries like Marshmallow) | Basic form validation                              | Built-in with Pydantic                           |

### Breakdown of Features:

- **Flask**:

  - **Pros**: Simple, minimalistic, and highly customizable. Great for small projects where you want to add only the components you need.
  - **Cons**: Requires more setup and extensions (e.g., Flask-RESTful, Flask-WTF) for basic functionality like form validation, authentication, and REST APIs. Limited asynchronous support.
  - **When to use**: Best for **small to medium projects**, quick prototyping, and applications where flexibility is more important than built-in features.
- **Django**:

  - **Pros**: A full-featured framework with built-in support for ORM, authentication, admin panel, and templating. Comes with **Django Rest Framework (DRF)** for building REST APIs.
  - **Cons**: Heavier and more opinionated than Flask. Somewhat slower performance and has limited asynchronous support. It can feel overwhelming for small projects due to its "batteries-included" approach.
  - **When to use**: Ideal for **large-scale applications** requiring lots of features (e.g., ORM, authentication, admin) out of the box. Use it when you need an **all-in-one solution**.
- **FastAPI**:

  - **Pros**: Designed for high-performance and asynchronous applications. **Built-in data validation** using Pydantic and **automatic API documentation** with Swagger UI and ReDoc. Excellent for real-time APIs or applications needing async I/O.
  - **Cons**: Smaller ecosystem compared to Django. It requires async programming for full performance.
  - **When to use**: Perfect for **high-performance APIs**, real-time applications (e.g., WebSockets), and **async-first projects**. Best for developers who need **speed and scalability** without heavy framework overhead.

### Summary:

- **Flask** is great for small, flexible projects that need minimal setup and customization.
- **Django** is a comprehensive, full-featured framework suited for complex, larger-scale applications where built-in features (e.g., ORM, admin) are critical.
- **FastAPI** is the go-to for high-performance, asynchronous applications, especially when building modern APIs or real-time applications.

```

```
