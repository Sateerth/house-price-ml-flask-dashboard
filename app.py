from flask import Flask, render_template, request, redirect, session, send_file, jsonify
import sqlite3
import pickle
import pandas as pd
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "secret123"

uploaded_data = None

# LOAD MODEL
try:
    model = pickle.load(open("model/house_model.pkl","rb"))
except:
    model = None

# DB INIT
def init_db():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        size INT,
        bedrooms INT,
        age INT,
        price REAL,
        time TEXT,
        user TEXT
    )
    """)

    conn.commit()
    conn.close()

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return redirect("/login")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            return "Missing input!"

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()

        conn.close()

        if user and check_password_hash(user[2], password):
            session["user"] = user[1]
            return redirect("/dashboard")
        else:
            return "Invalid login"

    return render_template("login.html")

# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET","POST"])
def signup():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            return "Missing input!"

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        hashed_password = generate_password_hash(password)

        try:
            cur.execute("INSERT INTO users (email,password) VALUES (?,?)",
                        (email, hashed_password))
            conn.commit()
        except:
            return "User already exists!"

        conn.close()

        return redirect("/login")

    return render_template("signup.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("SELECT * FROM predictions WHERE user=?", (session["user"],))
    data = cur.fetchall()

    conn.close()

    return render_template("dashboard.html", data=data)

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["GET","POST"])
def predict():

    if "user" not in session:
        return redirect("/login")

    global model

    if request.method == "POST":

        size = int(request.form["size"])
        bedrooms = int(request.form["bedrooms"])
        age = int(request.form["age"])

        if model is None:
            return "Upload dataset first!"
        
        scaler = pickle.load(open("model/scaler.pkl","rb"))

        input_data = pd.DataFrame([[size, bedrooms, age]],
                                columns=['size','bedrooms','age'])

        input_scaled = scaler.transform(input_data)

        price = int(round(model.predict(input_scaled)[0]))

        input_data = pd.DataFrame([[size, bedrooms, age]],
                                  columns=['size','bedrooms','age'])

        price = int(round(model.predict(input_data)[0]))

        time = datetime.now().strftime("%H:%M:%S")

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        cur.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                    (size, bedrooms, age, price, time, session["user"]))

        conn.commit()
        conn.close()

        return render_template("predict.html", prediction=price)

    return render_template("predict.html")

# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["GET","POST"])
def upload():

    global uploaded_data, model

    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":

        file = request.files["file"]
        path = "data/uploaded.csv"
        file.save(path)

        uploaded_data = pd.read_csv(path)

        X = uploaded_data[['size','bedrooms','age']]
        y = uploaded_data['price']

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X,y)

        pickle.dump(model, open("model/house_model.pkl","wb"))

        return render_template("upload.html",
                               preview=uploaded_data.head().to_html())

    return render_template("upload.html")

# ---------------- ANALYTICS ----------------
@app.route("/analytics")
def analytics():

    if "user" not in session:
        return redirect("/login")

    global uploaded_data

    if uploaded_data is None:
        return "Upload dataset first!"

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score

    X = uploaded_data[['size','bedrooms','age']]
    y = uploaded_data['price']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    lr = LinearRegression()
    dt = DecisionTreeRegressor()

    lr.fit(X_train,y_train)
    dt.fit(X_train,y_train)

    lr_pred = lr.predict(X_test)
    dt_pred = dt.predict(X_test)

    lr_acc = round(r2_score(y_test,lr_pred)*100,2)
    dt_acc = round(r2_score(y_test,dt_pred)*100,2)

    return render_template(
        "analytics.html",
        lr_acc=lr_acc,
        dt_acc=dt_acc,
        actual=y_test.tolist(),
        predicted=[int(x) for x in lr_pred]
    )

# ---------------- LIVE LOGS ----------------
@app.route("/get_logs")
def get_logs():

    if "user" not in session:
        return jsonify({"times": [], "prices": []})

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("""
        SELECT time, price 
        FROM predictions 
        WHERE user=? 
        ORDER BY ROWID DESC LIMIT 20
    """, (session["user"],))

    rows = cur.fetchall()
    conn.close()

    rows.reverse()

    return jsonify({
        "times": [row[0] for row in rows],
        "prices": [int(row[1]) for row in rows]
    })

# ---------------- DOWNLOAD CSV ----------------
@app.route("/download")
def download():

    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")

    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE user=?",
        conn,
        params=(session["user"],)
    )

    conn.close()

    path = "data/predictions.csv"
    df.to_csv(path,index=False)

    return send_file(path, as_attachment=True)

# ---------------- PDF ----------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import os
import yagmail
# ---------------- HEADER + FOOTER ----------------
def add_header_footer(canvas, doc):
    canvas.saveState()

    # HEADER
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(30, 800, "🏠 ML House Price Dashboard")

    # LOGO (if exists)
    if os.path.exists("static/logo.png"):
        try:
            canvas.drawImage("static/logo.png", 450, 780, width=100, height=40)
        except:
            pass

    # FOOTER
    canvas.setFont("Helvetica", 9)
    canvas.drawString(30, 30, f"Generated for: {session.get('user')}")
    canvas.drawRightString(550, 30, f"Page {doc.page}")

    canvas.restoreState()

# ---------------- WATERMARK ----------------
def add_watermark(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 50)
    canvas.setFillColorRGB(0.9, 0.9, 0.9)
    canvas.drawCentredString(300, 400, "ML DASHBOARD")
    canvas.restoreState()

@app.route("/download_pdf")
def download_pdf():

    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE user=?",
        conn,
        params=(session["user"],)
    )
    conn.close()

    if df.empty:
        return "No data available!"

    # ---------------- CHARTS ----------------
    plt.figure()
    plt.plot(df["price"])
    plt.title("Price Trend")
    plt.savefig("data/line.png")
    plt.close()

    plt.figure()
    plt.bar(range(len(df)), df["price"])
    plt.title("Price Distribution")
    plt.savefig("data/bar.png")
    plt.close()

    plt.figure()
    plt.pie(df["price"].head(5), autopct='%1.1f%%')
    plt.title("Top 5 Share")
    plt.savefig("data/pie.png")
    plt.close()

    # ---------------- PDF ----------------
    doc = SimpleDocTemplate(
        "report.pdf",
        pagesize=A4
    )

    styles = getSampleStyleSheet()
    content = []

    # ---------------- PAGE 1 (SUMMARY) ----------------
    content.append(Paragraph("<b>📊 Executive Summary</b>", styles['Title']))
    content.append(Spacer(1,20))

    avg_price = int(df["price"].mean())
    max_price = int(df["price"].max())
    min_price = int(df["price"].min())

    content.append(Paragraph(f"✔ Average Price: ₹ {avg_price}", styles['Normal']))
    content.append(Paragraph(f"✔ Maximum Price: ₹ {max_price}", styles['Normal']))
    content.append(Paragraph(f"✔ Minimum Price: ₹ {min_price}", styles['Normal']))

    trend = "Increasing 📈" if df["price"].iloc[-1] > df["price"].iloc[0] else "Decreasing 📉"

    content.append(Spacer(1,20))
    content.append(Paragraph("<b>🤖 Insights</b>", styles['Heading2']))
    content.append(Paragraph(f"• Market Trend: {trend}", styles['Normal']))
    content.append(Paragraph(f"• Total Predictions: {len(df)}", styles['Normal']))

    content.append(PageBreak())

    # ---------------- PAGE 2 (CHARTS) ----------------
    content.append(Paragraph("<b>📊 Visual Analytics</b>", styles['Title']))
    content.append(Spacer(1,20))

    content.append(Paragraph("Price Trend", styles['Heading2']))
    content.append(Image("data/line.png", width=400, height=200))

    content.append(Spacer(1,20))
    content.append(Paragraph("Distribution", styles['Heading2']))
    content.append(Image("data/bar.png", width=400, height=200))

    content.append(Spacer(1,20))
    content.append(Paragraph("Top Share", styles['Heading2']))
    content.append(Image("data/pie.png", width=400, height=200))

    content.append(PageBreak())

    # ---------------- PAGE 3 (TABLE) ----------------
    content.append(Paragraph("<b>📋 Prediction Data</b>", styles['Title']))
    content.append(Spacer(1,20))

    table_data = [["Size","Bedrooms","Age","Price","Time"]]

    for _, row in df.iterrows():
        table_data.append([
            row["size"],
            row["bedrooms"],
            row["age"],
            f"₹ {int(row['price'])}",
            row["time"]
        ])

    table = Table(table_data)

    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),1,colors.grey)
    ]))

    content.append(table)

    # ---------------- BUILD PDF ----------------
    doc.build(
        content,
        onFirstPage=lambda c, d: (add_header_footer(c,d), add_watermark(c,d)),
        onLaterPages=lambda c, d: (add_header_footer(c,d), add_watermark(c,d))
    )

    return send_file("report.pdf", as_attachment=True)

@app.route("/send_email")
def send_email():

    user_email = session.get("user")

    if not user_email:
        return redirect("/login")

    try:
        yag = yagmail.SMTP("your_email@gmail.com", "your_app_password")

        yag.send(
            to=user_email,
            subject="Your ML Prediction Report",
            contents="Attached is your dashboard report.",
            attachments="report.pdf"
        )

        return "Email sent successfully!"

    except Exception as e:
        return str(e)
    
    
# ---------------- RUN ----------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)