# main.py (FastAPI backend)

from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import pandas as pd
import numpy as np
import torch
from ecg import ECGModel, predict_ecg, generate_report
import io
import random
from fastapi.middleware.cors import CORSMiddleware

# FastAPI setup
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./heart_health.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)

class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    specialty = Column(String)
    rating = Column(Float)
    is_verified = Column(Integer)
    distance = Column(String)
    cases = Column(Integer)
    patients = Column(Integer)

class HealthData(Base):
    __tablename__ = "health_data"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    date = Column(String)
    heart_rate = Column(Integer)

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    author = Column(String)
    content = Column(String)
    likes = Column(Integer)
    comments = Column(Integer)
    type = Column(String)

Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str
    role: str

class UserOut(BaseModel):
    email: str
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ECGData(BaseModel):
    data: list[list[float]]

# Security setup
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Helper functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, email: str):
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db, email: str, password: str):
    user = get_user(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, email=email)
    if user is None:
        raise credentials_exception
    return user

# FastAPI routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=UserOut)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password, role=user.role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return UserOut(email=new_user.email, role=new_user.role)

@app.get("/users/me", response_model=UserOut)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return UserOut(email=current_user.email, role=current_user.role)

@app.get("/doctors")
async def get_doctors(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doctors = db.query(Doctor).all()
    return doctors

@app.get("/health-data")
async def get_health_data(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    data = db.query(HealthData).filter(HealthData.user_id == current_user.id).all()
    return data

@app.post("/health-data")
async def add_health_data(heart_rate: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_data = HealthData(user_id=current_user.id, date=datetime.now().strftime("%Y-%m-%d"), heart_rate=heart_rate)
    db.add(new_data)
    db.commit()
    db.refresh(new_data)
    return {"message": "Health data added successfully"}

@app.get("/posts")
async def get_posts(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    posts = db.query(Post).all()
    return posts

@app.post("/posts")
async def create_post(content: str, post_type: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_post = Post(author=current_user.email, content=content, likes=0, comments=0, type=post_type)
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    return {"message": "Post created successfully"}

@app.post("/analyze-ecg")
async def analyze_ecg(ecg_data: ECGData, current_user: User = Depends(get_current_user)):
    model_path = "ecg_model.pth"
    model = ECGModel(input_channels=3, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    ecg_np = np.array(ecg_data.data)
    prediction, confidence = predict_ecg(model, ecg_np)
    report = generate_report(prediction, confidence, ecg_np)

    return {"prediction": prediction, "confidence": confidence, "report": report}

@app.post("/predict")
async def predict(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    contents = await file.read()
    
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        ecg_data = df.values
        
        if ecg_data.shape[1] != 3:
            return {"error": "CSV should have exactly 3 columns for ECG channels."}
        
        model_path = "ecg_model.pth"
        model = ECGModel(input_channels=3, num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        prediction, confidence = predict_ecg(model, ecg_data)
        report = generate_report(prediction, confidence, ecg_data)
        
        return {"prediction": prediction, "confidence": confidence, "report": report}
    
    except Exception as e:
        return {"error": str(e)}

# Add sample data
def add_sample_data(db: Session):
    # Add a sample user
    if not get_user(db, "user@example.com"):
        hashed_password = get_password_hash("password123")
        db_user = User(email="user@example.com", hashed_password=hashed_password, role="client")
        db.add(db_user)

    # Add some sample doctors
    if db.query(Doctor).count() == 0:
        doctors = [
            Doctor(name="Dr. Smith", specialty="Cardiology", rating=4.5, is_verified=1, distance="2.5 miles", cases=150, patients=500),
            Doctor(name="Dr. Johnson", specialty="Internal Medicine", rating=4.2, is_verified=1, distance="3.7 miles", cases=200, patients=600),
            Doctor(name="Dr. Williams", specialty="Pediatrics", rating=4.8, is_verified=1, distance="1.8 miles", cases=180, patients=550),
        ]
        db.add_all(doctors)

    # Add some sample posts
    if db.query(Post).count() == 0:
        posts = [
            Post(author="Dr. Smith", content="New research on heart health...", likes=15, comments=3, type="text"),
            Post(author="Dr. Johnson", content="Tips for maintaining a healthy heart...", likes=20, comments=5, type="text"),
            Post(author="Dr. Williams", content="https://example.com/heart-health-article", likes=30, comments=8, type="article"),
            Post(author="Dr. Smith", content="https://example.com/heart-health-video", likes=25, comments=6, type="video"),
        ]
        db.add_all(posts)

    db.commit()

# Call this function to add sample data
with SessionLocal() as db:
    add_sample_data(db)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
