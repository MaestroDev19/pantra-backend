# Pantra Backend

This is the FastAPI backend service for **Pantra**, a smart pantry management and recipe/shopping assistant app.

It handles adding pantry items, computing AI embeddings via LangChain and Google Gemini, and storing/syncing them in the Supabase PostgreSQL database to support semantic similarity search (e.g. for matching ingredients to recipes).

## Tech Stack

- **Python 3.12+**
- **FastAPI** + **Uvicorn** (web server)
- **LangChain** + **Google GenAI** (for computing Gemini embeddings)
- **Supabase** (database client & vector store integration)
- **`uv`** (modern, ultra-fast Python package and project manager)

---

## Prerequisites

Make sure you have the following installed on your system:
- **Python 3.12** or higher
- **`uv`** package manager (recommended). To install `uv`, follow [Astral's official guide](https://docs.astral.sh/uv/getting-started/installation/):
  ```powershell
  # Windows Powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

---

## Configuration

The backend is configured using an `.env` file at the root of the `pantra-backend` directory.

Create or verify the `.env` file contains the following key configurations:

```env
# Supabase Configuration
SUPABASE_URL=https://<your-project-id>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
SUPABASE_PUBLISHABLE_KEY=<your-publishable-key>

# Google Generative AI / Gemini API
GOOGLE_GENERATIVE_AI_API_KEY=<your-gemini-api-key>

# Gemini Embeddings Model Settings
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.0
GEMINI_MAX_TOKENS=1000
GEMINI_MAX_RETRIES=2
GEMINI_EMBEDDINGS_MODEL=gemini-embedding-001
GEMINI_EMBEDDINGS_OUTPUT_DIMENSIONALITY=768
```

> [!IMPORTANT]
> The backend requires the **Supabase Service Role Key** (`SUPABASE_SERVICE_ROLE_KEY`) to bypass Row Level Security (RLS) policies for background embedding tasks and write operations. **Do not expose this key to the frontend client.**

---

## How to Run

### 1. Install Dependencies
Run the following command in the `pantra-backend` directory to automatically set up a virtual environment and install all required packages:
```bash
uv sync
```

### 2. Start the FastAPI Server
Launch the development server with reload enabled:
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Alternatively, you can run `main.py` directly:
```bash
uv run python main.py
```

The server will be running at:
- **API Base**: `http://localhost:8000/api`
- **Interactive Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **ReDoc Alternate Docs**: `http://localhost:8000/redoc`

---

## API Endpoints

### 1. Health Check
- **Endpoint**: `GET /api/health`
- **Description**: Verifies if the backend API service is running.

### 2. Add Pantry Item (Single)
- **Endpoint**: `POST /api/pantry/add`
- **Headers**:
  - `Authorization: Bearer <supabase_user_jwt>`
- **Request Body**:
  ```json
  {
    "name": "Organic Whole Milk",
    "category": "Dairy",
    "expiry_date": "2026-07-15",
    "household_id": "optional-household-uuid"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "id": "generated-pantry-item-uuid"
  }
  ```
- **Behavior**: Inserts the item immediately with `embedding_status = "pending"`. A background task computes the Gemini embedding and updates the vector in Supabase.

### 3. Add Pantry Items (Bulk)
- **Endpoint**: `POST /api/pantry/add/bulk`
- **Headers**:
  - `Authorization: Bearer <supabase_user_jwt>`
- **Request Body**:
  ```json
  [
    {
      "name": "Avocado",
      "category": "Produce",
      "expiry_date": "2026-07-05",
      "household_id": "optional-household-uuid"
    },
    {
      "name": "Greek Yogurt",
      "category": "Dairy",
      "expiry_date": "2026-07-20",
      "household_id": "optional-household-uuid"
    }
  ]
  ```

---

## Connecting to the Next.js Frontend

To route pantry addition requests from the frontend client to your local FastAPI backend server:

1. Open the frontend `.env` file located in the `pantra/` directory:
   ```env
   # pantra/.env
   NEXT_PUBLIC_PANTRY_API_URL=http://localhost:8000
   ```
2. When `NEXT_PUBLIC_PANTRY_API_URL` is defined, the frontend Next.js Server Actions will route all `addPantryItem` and `addPantryItemsBulk` calls through this backend API instead of inserting directly into Supabase.
3. If you want the frontend to fallback to direct database insert mode (bypassing the FastAPI backend), comment out or clear the `NEXT_PUBLIC_PANTRY_API_URL` variable:
   ```env
   # NEXT_PUBLIC_PANTRY_API_URL=
   ```

---

## Troubleshooting & Verification

### Test using `curl`
You can verify the backend is running and responds to health checks:
```bash
curl http://localhost:8000/api/health
```

To test adding an item via curl, obtain a valid user JWT token from your Supabase client session (or authentication page) and run:
```bash
curl -X POST http://localhost:8000/api/pantry/add \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_SUPABASE_USER_JWT>" \
  -d '{"name": "Apples", "category": "Produce"}'
```
Check the FastAPI terminal logs to see the background embedding computation task executing successfully.
