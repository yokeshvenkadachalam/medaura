from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from pydantic import BaseModel
from datetime import datetime, timedelta
import random

# ================= CONFIG =================

SECRET_KEY = "supersecret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
DATABASE_URL = "sqlite:///./medaura.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ================= MODELS =================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)


class Case(Base):
    __tablename__ = "cases"
    id = Column(Integer, primary_key=True)
    doctor_email = Column(String)
    symptoms = Column(String)
    ai_prediction = Column(String)
    doctor_decision = Column(String)
    doctor_confidence = Column(Integer)
    manual_mode = Column(Boolean)
    timestamp = Column(DateTime, default=datetime.utcnow)


class WeeklyTest(Base):
    __tablename__ = "weekly_tests"
    id = Column(Integer, primary_key=True)
    doctor_email = Column(String)
    score = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ================= SCHEMAS =================

class RegisterSchema(BaseModel):
    name: str
    email: str
    password: str


class LoginSchema(BaseModel):
    email: str
    password: str


class DiagnoseSchema(BaseModel):
    symptoms: str
    doctor_decision: str
    doctor_confidence: int


class WeeklySubmitSchema(BaseModel):
    score: int

# ================= UTILS =================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password):
    return pwd_context.hash(password)


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def create_token(email):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": email, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ================= AI ENGINE =================

def ai_predict(symptoms: str) -> str:
    s = symptoms.lower()

    if "fever" in s and "cough" in s:
        return "Flu"
    if "chest pain" in s:
        return "Cardiac Issue"
    if "headache" in s:
        return "Migraine"
    if "fatigue" in s:
        return "Anemia"

    return random.choice(["Flu", "Migraine", "Infection", "Viral Syndrome"])

# ================= APP =================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "MedAura Backend Running"}

# ================= AUTH =================

@app.post("/register")
def register(data: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = User(
        name=data.name,
        email=data.email,
        password=hash_password(data.password)
    )

    db.add(user)
    db.commit()
    return {"message": "Registered successfully"}


@app.post("/login")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()

    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user.email)
    return {"access_token": token}

# ================= DIAGNOSE =================

@app.post("/diagnose")
def diagnose(
    data: DiagnoseSchema,
    db: Session = Depends(get_db),
    email: str = Depends(get_current_user)
):
    ai_result = ai_predict(data.symptoms)

    is_match = data.doctor_decision.strip().lower() == ai_result.lower()

    suggestion = None
    if not is_match:
        suggestion = f"AI suggests reconsidering diagnosis as '{ai_result}'."

    total_cases = db.query(Case).filter(Case.doctor_email == email).count()
    manual_mode = True if (total_cases + 1) % 5 == 0 else False

    case = Case(
        doctor_email=email,
        symptoms=data.symptoms,
        ai_prediction=ai_result,
        doctor_decision=data.doctor_decision,
        doctor_confidence=data.doctor_confidence,
        manual_mode=manual_mode
    )

    db.add(case)
    db.commit()

    return {
        "ai_prediction": ai_result,
        "match_status": "Aligned" if is_match else "Mismatch",
        "suggestion": suggestion,
        "manual_mode": manual_mode
    }

# ================= DASHBOARD =================

@app.get("/dashboard")
def dashboard(
    db: Session = Depends(get_db),
    email: str = Depends(get_current_user)
):
    cases = db.query(Case).filter(Case.doctor_email == email).all()

    total = len(cases)
    agreement = sum(1 for c in cases if c.ai_prediction == c.doctor_decision)

    return {
        "total_cases": total,
        "agreement_rate": round((agreement / total) * 100, 2) if total else 0
    }

# ================= HISTORY =================

@app.get("/history")
def history(
    db: Session = Depends(get_db),
    email: str = Depends(get_current_user)
):
    cases = db.query(Case).filter(Case.doctor_email == email).all()

    return [
        {
            "symptoms": c.symptoms,
            "ai_prediction": c.ai_prediction,
            "doctor_decision": c.doctor_decision,
            "doctor_confidence": c.doctor_confidence,
            "manual_mode": c.manual_mode,
            "timestamp": c.timestamp
        }
        for c in cases
    ]

# ================= WEEKLY TEST =================

@app.get("/weekly")
def get_weekly_question():
    return {
        "question": "What is the normal body temperature?",
        "options": ["36-37°C", "40°C", "34°C", "42°C"]
    }

@app.post("/weekly")
def submit_weekly(
    data: WeeklySubmitSchema,
    db: Session = Depends(get_db),
    email: str = Depends(get_current_user)
):
    test = WeeklyTest(
        doctor_email=email,
        score=data.score
    )
    db.add(test)
    db.commit()
    return {"message": "Weekly test saved"}

@app.get("/weekly-score")
def get_weekly_score(
    db: Session = Depends(get_db),
    email: str = Depends(get_current_user)
):
    latest = (
        db.query(WeeklyTest)
        .filter(WeeklyTest.doctor_email == email)
        .order_by(WeeklyTest.timestamp.desc())
        .first()
    )

    if not latest:
        return {"score": None}

    return {"score": latest.score}
