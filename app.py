import os
import io
import base64
import json
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify, Response
)
from flask_mysqldb import MySQL
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import random

# Visualization imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --------------------
# App setup
# --------------------
app = Flask(__name__)
app.secret_key = "hr-analytics-secret-key-2025"

# MySQL Config - Update with your credentials
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "SSW@150607ssw"
app.config["MYSQL_DB"] = "hr_analytics"

mysql = MySQL(app)

# Flask-Login Config
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# Upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {"csv", "xlsx", "xls"}

# ML model paths
MODEL_FOLDER = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_FOLDER, exist_ok=True)
ATTITION_MODEL_PATH = os.path.join(MODEL_FOLDER, "attrition_model.pkl")
CLUSTERING_MODEL_PATH = os.path.join(MODEL_FOLDER, "clustering_model.pkl")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# --------------------
# User model
# --------------------
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        cur.close()
        if user:
            return User(int(user[0]), user[1], user[2])
    except Exception as e:
        print(f"Error loading user: {e}")
    return None

# --------------------
# Data Loading Utilities
# --------------------
def load_data_from_db():
    """Load data directly from MySQL database"""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM employees")
        result = cur.fetchall()
        cur.close()
        
        # Get column names
        cur = mysql.connection.cursor()
        cur.execute("SHOW COLUMNS FROM employees")
        columns = [col[0] for col in cur.fetchall()]
        cur.close()
        
        # Create DataFrame
        df = pd.DataFrame(result, columns=columns)
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return pd.DataFrame()

def load_data_from_session():
    """Load data from uploaded file in session"""
    if session.get("data_path"):
        try:
            data_path = session.get("data_path")
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            else:
                df = pd.read_excel(data_path)
            return df
        except Exception as e:
            print(f"Error loading data from session: {e}")
    return pd.DataFrame()

def get_available_data():
    """Get available data from either database or session"""
    # Try to load from database first
    df = load_data_from_db()
    if not df.empty:
        return df
    
    # Fall back to session data
    return load_data_from_session()

# --------------------
# ML Model Training
# --------------------
def train_attrition_model(df):
    """Train and save an attrition prediction model"""
    try:
        # Prepare data for ML
        ml_df = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                           'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        label_encoders = {}
        
        for col in categorical_cols:
            if col in ml_df.columns:
                le = LabelEncoder()
                ml_df[col] = le.fit_transform(ml_df[col].astype(str))
                label_encoders[col] = le
        
        # Convert target variable
        if 'Attrition' in ml_df.columns:
            ml_df['Attrition'] = ml_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Select features and target
        feature_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 
                       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                       'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
                       'NumCompaniesWorked', 'PercentSalaryHike', 
                       'PerformanceRating', 'RelationshipSatisfaction',
                       'StockOptionLevel', 'TotalWorkingYears', 
                       'TrainingTimesLastYear', 'WorkLifeBalance',
                       'YearsAtCompany', 'YearsInCurrentRole', 
                       'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        # Only use columns that exist in the dataframe
        available_features = [col for col in feature_cols if col in ml_df.columns]
        available_features.extend([col for col in categorical_cols if col in ml_df.columns])
        
        X = ml_df[available_features]
        y = ml_df['Attrition'] if 'Attrition' in ml_df.columns else None
        
        if y is None or len(available_features) == 0:
            return None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        # Save model
        joblib.dump(model, ATTITION_MODEL_PATH)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, feature_importance
        
    except Exception as e:
        print(f"Error training attrition model: {e}")
        return None, None

def perform_clustering(df):
    """Perform employee clustering using K-means"""
    try:
        # Prepare data for clustering
        cluster_df = df.copy()
        
        # Select numerical features
        numerical_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 
                             'YearsAtCompany', 'PerformanceRating', 'JobSatisfaction']
        
        # Only use columns that exist
        available_features = [col for col in numerical_features if col in cluster_df.columns]
        
        if len(available_features) < 2:
            return None, None, None
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df[available_features].fillna(0))
        
        # Determine optimal k using elbow method
        wcss = []
        max_clusters = min(10, len(scaled_data) - 1)
        
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)
        
        # Use elbow method to find optimal k (simplified)
        optimal_k = 3  # Default to 3 clusters
        
        # Apply K-means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Save model
        joblib.dump(kmeans, CLUSTERING_MODEL_PATH)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create cluster profiles
        cluster_df['Cluster'] = clusters
        cluster_profiles = cluster_df.groupby('Cluster')[available_features].mean()
        
        return clusters, pca_result, cluster_profiles
        
    except Exception as e:
        print(f"Error performing clustering: {e}")
        return None, None, None

# --------------------
# Visualization utilities
# --------------------
def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plotly_to_html(fig):
    """Convert plotly figure to HTML string"""
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

def find_similar_column(df, target_columns):
    """Find similar columns in the dataframe"""
    available_columns = df.columns.str.lower().tolist()
    for target in target_columns:
        target_lower = target.lower()
        if target_lower in available_columns:
            return target
        # Check for partial matches
        for col in available_columns:
            if target_lower in col or col in target_lower:
                return col
    return None

def build_visuals(df, filters=None):
    """Generate all visualizations from dataframe with optional filters"""
    # Apply filters if provided
    if filters:
        filtered_df = df.copy()
        
        # Department filter
        if filters.get('department') and filters['department'] != 'All':
            filtered_df = filtered_df[filtered_df['Department'] == filters['department']]
        
        # Job Role filter
        if filters.get('job_role') and filters['job_role'] != 'All':
            filtered_df = filtered_df[filtered_df['JobRole'] == filters['job_role']]
        
        # Salary range filter
        if filters.get('min_salary') and filters.get('max_salary'):
            min_sal = float(filters['min_salary'])
            max_sal = float(filters['max_salary'])
            filtered_df = filtered_df[
                (filtered_df['MonthlyIncome'] >= min_sal) & 
                (filtered_df['MonthlyIncome'] <= max_sal)
            ]
    else:
        filtered_df = df
    
    visuals = {}
    available_columns = filtered_df.columns.tolist()
    column_mapping = {}
    
    # Map expected columns to available columns
    expected_columns = {
        "Age": ["age", "employee age", "staff age"],
        "MonthlyIncome": ["monthlyincome", "salary", "income", "monthly income", "monthly salary"],
        "TotalWorkingYears": ["totalworkingyears", "experience", "years of experience", "work experience"],
        "Department": ["department", "dept", "division"],
        "JobRole": ["jobrole", "role", "position", "job title"],
        "PerformanceRating": ["performancerating", "performance", "rating", "performance score"],
        "Attrition": ["attrition", "left company", "terminated", "resigned"],
        "JobSatisfaction": ["jobsatisfaction", "satisfaction", "job satisfaction"],
        "Gender": ["gender", "sex"],
        "YearsAtCompany": ["yearsatcompany", "tenure", "company tenure", "years in company"]
    }
    
    # Create column mapping
    for expected_col, alternatives in expected_columns.items():
        found_col = find_similar_column(filtered_df, [expected_col] + alternatives)
        if found_col:
            column_mapping[expected_col] = found_col
    
    # Set style for all static plots
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution plots
    distribution_columns = [
        ("Age", "dist_age_png"),
        ("MonthlyIncome", "dist_income_png"),
        ("TotalWorkingYears", "dist_experience_png")
    ]
    
    for expected_col, key in distribution_columns:
        if expected_col in column_mapping:
            actual_col = column_mapping[expected_col]
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(filtered_df[actual_col].dropna(), kde=True, ax=ax, bins=20)
                ax.set_title(f"Distribution of {actual_col}")
                visuals[key] = fig_to_base64(fig)
            except Exception as e:
                print(f"Distribution plot error for {actual_col}: {e}")
    
    # 2. Department-wise performance heatmap
    if ("Department" in column_mapping and "JobRole" in column_mapping and 
        "PerformanceRating" in column_mapping):
        try:
            dept_col = column_mapping["Department"]
            role_col = column_mapping["JobRole"]
            perf_col = column_mapping["PerformanceRating"]
            
            pivot = pd.pivot_table(
                filtered_df,
                values=perf_col,
                index=dept_col,
                columns=role_col,
                aggfunc=np.mean,
                fill_value=0
            )
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=True)
                ax.set_title(f"{dept_col} vs {role_col} — Avg {perf_col}")
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                visuals["heatmap_dept_perf_png"] = fig_to_base64(fig)
        except Exception as e:
            print(f"Heatmap error: {e}")
    
    # 3. Attrition vs Job Satisfaction
    if ("Attrition" in column_mapping and "JobSatisfaction" in column_mapping):
        try:
            attrition_col = column_mapping["Attrition"]
            satisfaction_col = column_mapping["JobSatisfaction"]
            
            group = filtered_df.groupby(attrition_col)[satisfaction_col].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=group, x=attrition_col, y=satisfaction_col, ax=ax)
            ax.set_title(f"{attrition_col} vs {satisfaction_col} (mean)")
            visuals["attrition_jsat_png"] = fig_to_base64(fig)
        except Exception as e:
            print(f"Attrition plot error: {e}")
    
    # 4. Correlation heatmap
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        try:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, annot=True, fmt=".2f")
            ax.set_title("Correlation Heatmap (numeric features)")
            visuals["corr_heatmap_png"] = fig_to_base64(fig)
        except Exception as e:
            print(f"Correlation heatmap error: {e}")
    
    # 5. Boxplot: MonthlyIncome vs PerformanceRating
    if ("MonthlyIncome" in column_mapping and "PerformanceRating" in column_mapping):
        try:
            income_col = column_mapping["MonthlyIncome"]
            perf_col = column_mapping["PerformanceRating"]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=filtered_df, x=perf_col, y=income_col, ax=ax)
            ax.set_title(f"{income_col} by {perf_col}")
            visuals["box_income_perf_png"] = fig_to_base64(fig)
        except Exception as e:
            print(f"Boxplot error: {e}")
    
    # 6. Pie chart: Gender Ratio
    if "Gender" in column_mapping:
        try:
            gender_col = column_mapping["Gender"]
            counts = filtered_df[gender_col].value_counts(dropna=False)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"{gender_col} Ratio")
            ax.axis("equal")
            visuals["pie_gender_png"] = fig_to_base64(fig)
        except Exception as e:
            print(f"Pie chart error: {e}")
    
    # 7. Interactive: Plotly bar — Department Performance
    if ("Department" in column_mapping and "PerformanceRating" in column_mapping):
        try:
            dept_col = column_mapping["Department"]
            perf_col = column_mapping["PerformanceRating"]
            
            dep_perf = filtered_df.groupby(dept_col)[perf_col].mean().sort_values(ascending=False)
            fig_bar = px.bar(
                x=dep_perf.index.tolist(),
                y=dep_perf.values.tolist(),
                title=f"Average {perf_col} by {dept_col}",
                labels={"x": dept_col, "y": f"Average {perf_col}"}
            )
            fig_bar.update_traces(texttemplate='%{y:.2f}', textposition='outside')
            visuals["dept_perf_bar_html"] = plotly_to_html(fig_bar)
        except Exception as e:
            print(f"Plotly bar error: {e}")
    
    # 8. Interactive: Plotly trend — YearsAtCompany vs PerformanceRating
    if ("YearsAtCompany" in column_mapping and "PerformanceRating" in column_mapping):
        try:
            years_col = column_mapping["YearsAtCompany"]
            perf_col = column_mapping["PerformanceRating"]
            
            # Clean data
            clean_df = filtered_df[[years_col, perf_col]].dropna()
            
            if len(clean_df) > 1:
                fig_trend = px.scatter(
                    clean_df, x=years_col, y=perf_col,
                    title=f"{years_col} vs {perf_col}",
                    trendline="lowess"
                )
                visuals["years_perf_trend_html"] = plotly_to_html(fig_trend)
        except Exception as e:
            print(f"Plotly trend error: {e}")
    
    # 1. Interactive Scatter Plot: Salary vs Experience by Department
    if ("MonthlyIncome" in column_mapping and "TotalWorkingYears" in column_mapping and 
        "Department" in column_mapping):
        try:
            income_col = column_mapping["MonthlyIncome"]
            exp_col = column_mapping["TotalWorkingYears"]
            dept_col = column_mapping["Department"]
            
            fig_scatter = px.scatter(
                filtered_df, 
                x=exp_col, 
                y=income_col, 
                color=dept_col,
                hover_data=['EmployeeNumber', 'JobRole'] if 'JobRole' in filtered_df.columns else None,
                title=f"{income_col} vs {exp_col} by {dept_col}",
                labels={exp_col: "Years of Experience", income_col: "Monthly Income ($)"}
            )
            visuals["scatter_salary_exp_html"] = plotly_to_html(fig_scatter)
        except Exception as e:
            print(f"Scatter plot error: {e}")
    
    # 2. Interactive Sunburst Chart: Department > Job Role > Gender
    if ("Department" in column_mapping and "JobRole" in column_mapping and 
        "Gender" in column_mapping and "MonthlyIncome" in column_mapping):
        try:
            dept_col = column_mapping["Department"]
            role_col = column_mapping["JobRole"]
            gender_col = column_mapping["Gender"]
            income_col = column_mapping["MonthlyIncome"]
            
            # Create aggregated data for sunburst
            sunburst_data = filtered_df.groupby([dept_col, role_col, gender_col]).agg({
                income_col: 'mean',
                'EmployeeNumber': 'count'
            }).reset_index()
            
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=[dept_col, role_col, gender_col],
                values='EmployeeNumber',
                color=income_col,
                color_continuous_scale='Viridis',
                title="Organization Hierarchy: Department > Job Role > Gender",
                hover_data={income_col: ':.2f'}
            )
            visuals["sunburst_org_html"] = plotly_to_html(fig_sunburst)
        except Exception as e:
            print(f"Sunburst chart error: {e}")
    
    # 3. Interactive Box Plot: Salary Distribution by Job Role
    if ("MonthlyIncome" in column_mapping and "JobRole" in column_mapping):
        try:
            income_col = column_mapping["MonthlyIncome"]
            role_col = column_mapping["JobRole"]
            
            fig_box = px.box(
                filtered_df, 
                x=role_col, 
                y=income_col,
                color=role_col,
                title=f"{income_col} Distribution by {role_col}",
                labels={income_col: "Monthly Income ($)", role_col: "Job Role"}
            )
            fig_box.update_layout(xaxis_tickangle=45)
            visuals["box_salary_role_html"] = plotly_to_html(fig_box)
        except Exception as e:
            print(f"Box plot error: {e}")
    
    # 4. Interactive Bar Chart: Average Metrics by Department
    if ("Department" in column_mapping and "MonthlyIncome" in column_mapping and 
        "PerformanceRating" in column_mapping and "JobSatisfaction" in column_mapping):
        try:
            dept_col = column_mapping["Department"]
            income_col = column_mapping["MonthlyIncome"]
            perf_col = column_mapping["PerformanceRating"]
            satisfaction_col = column_mapping["JobSatisfaction"]
            
            dept_metrics = filtered_df.groupby(dept_col).agg({
                income_col: 'mean',
                perf_col: 'mean',
                satisfaction_col: 'mean'
            }).reset_index()
            
            fig_bar = px.bar(
                dept_metrics, 
                x=dept_col, 
                y=[income_col, perf_col, satisfaction_col],
                title="Average Metrics by Department",
                barmode='group',
                labels={'value': 'Average Value', 'variable': 'Metric'}
            )
            visuals["dept_metrics_bar_html"] = plotly_to_html(fig_bar)
        except Exception as e:
            print(f"Department metrics bar error: {e}")
    
    # 5. Interactive Pie Chart: Attrition Rate
    if "Attrition" in column_mapping:
        try:
            attrition_col = column_mapping["Attrition"]
            attrition_counts = filtered_df[attrition_col].value_counts()
            
            fig_pie = px.pie(
                values=attrition_counts.values,
                names=attrition_counts.index,
                title="Attrition Distribution",
                hole=0.4
            )
            visuals["attrition_pie_html"] = plotly_to_html(fig_pie)
        except Exception as e:
            print(f"Attrition pie chart error: {e}")
    
    # 6. Interactive Histogram: Age Distribution with Filters
    if "Age" in column_mapping and "Gender" in column_mapping:
        try:
            age_col = column_mapping["Age"]
            gender_col = column_mapping["Gender"]
            
            fig_hist = px.histogram(
                filtered_df, 
                x=age_col, 
                color=gender_col,
                marginal="box",
                title="Age Distribution by Gender",
                nbins=20,
                barmode="overlay"
            )
            visuals["age_dist_hist_html"] = plotly_to_html(fig_hist)
        except Exception as e:
            print(f"Age histogram error: {e}")
    
    # Add column mapping to visuals for debugging
    visuals["column_mapping"] = column_mapping
    
    return visuals

# --------------------
# Routes
# --------------------
@app.route("/export/<format>")
@login_required
def export_data(format):
    """Export data in various formats"""
    df = get_available_data()
    if df.empty:
        flash("No data available for export", "warning")
        return redirect(url_for("visualize"))
    
    try:
        if format == 'csv':
            # Export as CSV
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return Response(
                output,
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment;filename=hr_data_export.csv"}
            )
            
        elif format == 'excel':
            # Export as Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Employee Data')
            output.seek(0)
            
            return Response(
                output,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment;filename=hr_data_export.xlsx"}
            )
            
        else:
            flash("Unsupported export format", "danger")
            return redirect(url_for("visualize"))
            
    except Exception as e:
        flash(f"Error exporting data: {e}", "danger")
        return redirect(url_for("visualize"))
@app.route("/")
def index():
    if current_user.is_authenticated and session.get("data_path"):
        return redirect(url_for("visualize"))
    return redirect(url_for("login"))
@app.route("/api/employee_data")
@login_required
def api_employee_data():
    df = get_available_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404
    
    # Return sample data for API
    sample_data = df.head(50).to_dict('records')
    return jsonify(sample_data)

@app.route("/api/department_stats")
@login_required
def api_department_stats():
    df = get_available_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404
    
    # Calculate department statistics
    if 'Department' in df.columns:
        dept_stats = df.groupby('Department').agg({
            'MonthlyIncome': ['count', 'mean', 'median'],
            'Age': 'mean',
            'PerformanceRating': 'mean'
        }).round(2).to_dict()
        
        return jsonify(dept_stats)
    else:
        return jsonify({"error": "Department data not available"}), 400 
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("visualize"))
        
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            flash("Please enter both username and password", "danger")
            return render_template("login.html")

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            cur.close()

            if user and check_password_hash(user[2], password):
                user_obj = User(int(user[0]), user[1], user[2])
                login_user(user_obj)
                flash("Login successful!", "success")
                return redirect(url_for("visualize"))
            else:
                flash("Invalid username or password", "danger")
        except Exception as e:
            flash("Database error. Please try again.", "danger")
            print(f"Login error: {e}")
    
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("visualize"))
        
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        if not username or not password:
            flash("Please enter both username and password", "danger")
            return render_template("register.html")
        
        if len(password) < 6:
            flash("Password must be at least 6 characters long", "warning")
            return render_template("register.html")
        
        # Use the correct method format for password hashing
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        try:
            cur = mysql.connection.cursor()
            
            # Check if username already exists
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                flash("Username already exists", "danger")
                cur.close()
                return render_template("register.html")
                
            # Create new user
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", 
                       (username, hashed_password))
            mysql.connection.commit()
            cur.close()
            
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash("Registration failed. Please try again.", "danger")
            print(f"Registration error: {e}")
    
    return render_template("register.html")

@app.route("/dashboard")
@login_required
def dashboard():
    # Redirect to visualize page
    return redirect(url_for("visualize"))

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in request.", "danger")
            return redirect(url_for("upload"))
            
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "warning")
            return redirect(url_for("upload"))
            
        if not allowed_file(file.filename):
            flash("Unsupported file type. Upload CSV or Excel.", "warning")
            return redirect(url_for("upload"))

        try:
            # Save file
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            session["data_path"] = save_path
            
            # Test if file can be read
            if file.filename.endswith('.csv'):
                test_df = pd.read_csv(save_path)
            else:
                test_df = pd.read_excel(save_path)
                
            flash("File uploaded successfully.", "success")
            return redirect(url_for("visualize"))
        except Exception as e:
            flash(f"Error reading file: {str(e)}", "danger")
            if os.path.exists(save_path):
                os.remove(save_path)
            return redirect(url_for("upload"))
    
    return render_template("upload.html")

@app.route("/visualize", methods=["GET", "POST"])
@login_required
def visualize():
    # Get filter parameters
    department = request.args.get('department', 'All')
    job_role = request.args.get('job_role', 'All')
    min_salary = request.args.get('min_salary', 0)
    max_salary = request.args.get('max_salary', 100000)
    
    filters = {
        'department': department,
        'job_role': job_role,
        'min_salary': min_salary,
        'max_salary': max_salary
    }
    
    # Handle file upload directly from visualize page
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in request.", "danger")
            return render_template("visualize.html")
            
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "warning")
            return render_template("visualize.html")
            
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            flash("Only CSV or Excel files are supported.", "warning")
            return render_template("visualize.html")

        try:
            # Read file
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
                
            if df.empty:
                flash("The uploaded file is empty.", "warning")
                return render_template("visualize.html")
                
            # Save file to session
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            session["data_path"] = save_path
            session["df_columns"] = df.columns.tolist()  # Store column names for reference
            
            # Generate visualizations
            visuals = build_visuals(df, filters)
            n_rows, n_cols = df.shape
            
            # Show which columns were detected
            if "column_mapping" in visuals:
                detected_columns = visuals.pop("column_mapping")
                flash(f"Detected columns: {', '.join([f'{k}→{v}' for k, v in detected_columns.items()])}", "info")
            
            return render_template(
                "visualize.html",
                n_rows=n_rows,
                n_cols=n_cols,
                filters=filters,
                **visuals
            )
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "danger")
            return render_template("visualize.html")
    
    # GET request - check if data already exists
    df = get_available_data()
    if not df.empty:
        try:
            # Get unique values for filters
            departments = ['All'] + sorted(df['Department'].dropna().unique().tolist()) if 'Department' in df.columns else ['All']
            job_roles = ['All'] + sorted(df['JobRole'].dropna().unique().tolist()) if 'JobRole' in df.columns else ['All']
            
            # Generate visualizations
            visuals = build_visuals(df, filters)
            n_rows, n_cols = df.shape
            
            # Show which columns were detected
            if "column_mapping" in visuals:
                detected_columns = visuals.pop("column_mapping")
                flash(f"Detected columns: {', '.join([f'{k}→{v}' for k, v in detected_columns.items()])}", "info")
            
            return render_template(
                "visualize.html",
                n_rows=n_rows,
                n_cols=n_cols,
                departments=departments,
                job_roles=job_roles,
                filters=filters,
                **visuals
            )
        except Exception as e:
            flash(f"Error loading data: {e}", "danger")
    
    # No data available, show empty page with filters disabled
    return render_template("visualize.html", departments=[], job_roles=[])

@app.route("/attrition_prediction", methods=["GET", "POST"])
@login_required
def attrition_prediction():
    df = get_available_data()
    if df.empty:
        flash("No data available for analysis", "warning")
        return redirect(url_for("visualize"))
    
    # Train or load model
    if os.path.exists(ATTITION_MODEL_PATH):
        model = joblib.load(ATTITION_MODEL_PATH)
        feature_importance = None  # Would need to be stored separately
    else:
        model, feature_importance = train_attrition_model(df)
    
    if model is None:
        flash("Could not train prediction model. Check your data.", "danger")
        return redirect(url_for("visualize"))
    
    # Make predictions
    try:
        # Prepare data for prediction
        ml_df = df.copy()
        
        # Encode categorical variables (simplified approach)
        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                           'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        
        for col in categorical_cols:
            if col in ml_df.columns:
                le = LabelEncoder()
                ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        
        # Select features
        feature_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 
                       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                       'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
                       'NumCompaniesWorked', 'PercentSalaryHike', 
                       'PerformanceRating', 'RelationshipSatisfaction',
                       'StockOptionLevel', 'TotalWorkingYears', 
                       'TrainingTimesLastYear', 'WorkLifeBalance',
                       'YearsAtCompany', 'YearsInCurrentRole', 
                       'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        # Only use columns that exist in the dataframe and model was trained on
        available_features = [col for col in feature_cols if col in ml_df.columns and hasattr(model, 'feature_importances_')]
        available_features.extend([col for col in categorical_cols if col in ml_df.columns and hasattr(model, 'feature_importances_')])
        
        # Make predictions
        predictions = model.predict_proba(ml_df[available_features])[:, 1]  # Probability of attrition
        df['AttritionRisk'] = predictions
        
        # Create risk categories
        df['RiskCategory'] = pd.cut(predictions, 
                                   bins=[0, 0.3, 0.7, 1], 
                                   labels=['Low', 'Medium', 'High'])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        risk_counts = df['RiskCategory'].value_counts()
        ax.bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
        ax.set_title('Employee Attrition Risk Distribution')
        ax.set_ylabel('Number of Employees')
        risk_chart = fig_to_base64(fig)
        
        # Feature importance visualization
        if feature_importance is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(10)
            ax.barh(top_features['feature'], top_features['importance'])
            ax.set_title('Top Features Influencing Attrition')
            ax.set_xlabel('Importance')
            feature_importance_chart = fig_to_base64(fig)
        else:
            feature_importance_chart = None
        
        # Get high-risk employees
        high_risk_employees = df[df['RiskCategory'] == 'High'][['EmployeeNumber', 'Age', 'Department', 
                                                               'JobRole', 'MonthlyIncome', 'AttritionRisk']]
        
        return render_template(
            "attrition_prediction.html",
            risk_chart=risk_chart,
            feature_importance_chart=feature_importance_chart,
            high_risk_employees=high_risk_employees.to_dict('records')
        )
        
    except Exception as e:
        flash(f"Error making predictions: {e}", "danger")
        return redirect(url_for("visualize"))

@app.route("/employee_clustering", methods=["GET", "POST"])
@login_required
def employee_clustering():
    df = get_available_data()
    if df.empty:
        flash("No data available for analysis", "warning")
        return redirect(url_for("visualize"))
    
    # Perform clustering
    clusters, pca_result, cluster_profiles = perform_clustering(df)
    
    if clusters is None:
        flash("Could not perform clustering. Check your data.", "danger")
        return redirect(url_for("visualize"))
    
    # Add clusters to dataframe
    df['Cluster'] = clusters
    
    # Create cluster visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    ax.set_title('Employee Clusters (PCA Visualization)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.colorbar(scatter)
    cluster_viz = fig_to_base64(fig)
    
    # Create cluster profiles visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_means = df.groupby('Cluster')[['Age', 'MonthlyIncome', 'PerformanceRating', 'JobSatisfaction']].mean()
    cluster_means.plot(kind='bar', ax=ax)
    ax.set_title('Cluster Profiles')
    ax.set_ylabel('Average Value')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    profile_viz = fig_to_base64(fig)
    
    # Describe each cluster
    cluster_descriptions = {}
    for cluster_id in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster_id]
        description = {
            'size': len(cluster_data),
            'avg_age': cluster_data['Age'].mean(),
            'avg_income': cluster_data['MonthlyIncome'].mean(),
            'avg_performance': cluster_data['PerformanceRating'].mean() if 'PerformanceRating' in cluster_data.columns else None,
            'common_department': cluster_data['Department'].mode()[0] if 'Department' in cluster_data.columns else None,
            'common_job_role': cluster_data['JobRole'].mode()[0] if 'JobRole' in cluster_data.columns else None
        }
        cluster_descriptions[cluster_id] = description
    
    return render_template(
        "employee_clustering.html",
        cluster_viz=cluster_viz,
        profile_viz=profile_viz,
        cluster_descriptions=cluster_descriptions,
        clusters=df['Cluster'].value_counts().to_dict()
    )

@app.route("/time_series_analysis")
@login_required
def time_series_analysis():
    df = get_available_data()
    if df.empty:
        flash("No data available for analysis", "warning")
        return redirect(url_for("visualize"))
    
    # Generate synthetic time series data (since your schema doesn't have dates)
    # In a real scenario, you would have hire dates, termination dates, etc.
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    
    # Simulate hiring trends
    hire_trend = np.sin(np.arange(len(dates)) * 0.5) + np.random.normal(0, 0.2, len(dates)) + 1
    hires = np.round(hire_trend * 10).astype(int)
    
    # Simulate attrition trends with seasonality
    attrition_trend = np.sin(np.arange(len(dates)) * 0.8 + 2) + np.random.normal(0, 0.3, len(dates)) + 0.5
    attritions = np.round(attrition_trend * 5).astype(int)
    
    # Simulate performance trends
    performance_trend = np.cos(np.arange(len(dates)) * 0.3) + np.random.normal(0, 0.1, len(dates)) + 3.5
    
    # Create time series dataframe
    ts_df = pd.DataFrame({
        'Date': dates,
        'Hires': hires,
        'Attritions': attritions,
        'Performance': performance_trend
    })
    
    # Create visualizations
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(ts_df['Date'], ts_df['Hires'], label='Hires', marker='o')
    ax.plot(ts_df['Date'], ts_df['Attritions'], label='Attritions', marker='s')
    ax.set_title('Hiring and Attrition Trends Over Time')
    ax.set_ylabel('Count')
    ax.legend()
    plt.xticks(rotation=45)
    hires_attrition_chart = fig_to_base64(fig)
    
    # Performance trend
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts_df['Date'], ts_df['Performance'], label='Performance Rating', marker='^', color='green')
    ax.set_title('Average Performance Rating Over Time')
    ax.set_ylabel('Performance Rating (1-5)')
    ax.set_ylim(1, 5)
    plt.xticks(rotation=45)
    performance_chart = fig_to_base64(fig)
    
    # Calculate key metrics
    total_hires = ts_df['Hires'].sum()
    total_attritions = ts_df['Attritions'].sum()
    avg_performance = ts_df['Performance'].mean()
    net_growth = total_hires - total_attritions
    
    # Monthly statistics
    monthly_stats = ts_df.describe().to_dict()
    
    return render_template(
        "time_series_analysis.html",
        hires_attrition_chart=hires_attrition_chart,
        performance_chart=performance_chart,
        total_hires=total_hires,
        total_attritions=total_attritions,
        net_growth=net_growth,
        avg_performance=avg_performance,
        monthly_stats=monthly_stats
    )

@app.route("/data_insights")
@login_required
def data_insights():
    df = get_available_data()
    if df.empty:
        flash("No data available for analysis", "warning")
        return redirect(url_for("visualize"))
    
    insights = []
    
    # Basic statistics insights
    if 'MonthlyIncome' in df.columns:
        avg_income = df['MonthlyIncome'].mean()
        max_income = df['MonthlyIncome'].max()
        min_income = df['MonthlyIncome'].min()
        insights.append(f"Salary Range: ${min_income:,.0f} - ${max_income:,.0f} (Avg: ${avg_income:,.0f})")
    
    if 'Age' in df.columns:
        avg_age = df['Age'].mean()
        insights.append(f"Average Employee Age: {avg_age:.1f} years")
    
    if 'Department' in df.columns and 'Gender' in df.columns:
        dept_gender = df.groupby(['Department', 'Gender']).size().unstack(fill_value=0)
        for dept in dept_gender.index:
            if 'Male' in dept_gender.columns and 'Female' in dept_gender.columns:
                male_count = dept_gender.loc[dept, 'Male']
                female_count = dept_gender.loc[dept, 'Female']
                total = male_count + female_count
                if total > 0:
                    ratio = female_count / total * 100
                    insights.append(f"{dept}: {ratio:.1f}% female employees")
    
    if 'Attrition' in df.columns:
        attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
        insights.append(f"Overall Attrition Rate: {attrition_rate:.1f}%")
    
    if 'JobSatisfaction' in df.columns:
        avg_satisfaction = df['JobSatisfaction'].mean()
        insights.append(f"Average Job Satisfaction: {avg_satisfaction:.1f}/4")
    
    if 'PerformanceRating' in df.columns:
        avg_performance = df['PerformanceRating'].mean()
        insights.append(f"Average Performance Rating: {avg_performance:.1f}/5")
    
    # Correlation insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        if 'MonthlyIncome' in numeric_cols:
            income_corr = corr_matrix['MonthlyIncome'].drop('MonthlyIncome').abs().sort_values(ascending=False)
            if len(income_corr) > 0:
                top_corr = income_corr.index[0]
                insights.append(f"Salary correlates most strongly with {top_corr} (r={income_corr.iloc[0]:.2f})")
    
    return render_template("data_insights.html", insights=insights)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for("login"))

@app.route("/download_report")
@login_required
def download_report():
    df = get_available_data()
    if df.empty:
        flash("No data available for report", "warning")
        return redirect(url_for("visualize"))
    
    # Create a simple report
    report_content = f"HR Analytics Report\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_content += f"Total Employees: {len(df)}\n"
    
    if 'Department' in df.columns:
        report_content += "\nDepartment Distribution:\n"
        dept_counts = df['Department'].value_counts()
        for dept, count in dept_counts.items():
            report_content += f"  {dept}: {count} employees\n"
    
    if 'MonthlyIncome' in df.columns:
        report_content += f"\nSalary Statistics:\n"
        report_content += f"  Average: ${df['MonthlyIncome'].mean():,.0f}\n"
        report_content += f"  Minimum: ${df['MonthlyIncome'].min():,.0f}\n"
        report_content += f"  Maximum: ${df['MonthlyIncome'].max():,.0f}\n"
    
    # Create response
    response = Response(report_content, mimetype="text/plain")
    response.headers["Content-Disposition"] = "attachment; filename=hr_report.txt"
    return response

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    # Create database tables if they don't exist
    try:
        cur = mysql.connection.cursor()
        
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(80) UNIQUE NOT NULL,
                password VARCHAR(120) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create employees table (simplified version)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                EmployeeNumber INT PRIMARY KEY,
                Age INT,
                Attrition VARCHAR(10),
                BusinessTravel VARCHAR(50),
                DailyRate INT,
                Department VARCHAR(50),
                DistanceFromHome INT,
                Education INT,
                EducationField VARCHAR(50),
                EmployeeCount INT,
                EnvironmentSatisfaction INT,
                Gender VARCHAR(10),
                HourlyRate INT,
                JobInvolvement INT,
                JobLevel INT,
                JobRole VARCHAR(50),
                JobSatisfaction INT,
                MaritalStatus VARCHAR(20),
                MonthlyIncome INT,
                MonthlyRate INT,
                NumCompaniesWorked INT,
                OverTime VARCHAR(10),
                PercentSalaryHike INT,
                PerformanceRating INT,
                RelationshipSatisfaction INT,
                StandardHours INT,
                StockOptionLevel INT,
                TotalWorkingYears INT,
                TrainingTimesLastYear INT,
                WorkLifeBalance INT,
                YearsAtCompany INT,
                YearsInCurrentRole INT,
                YearsSinceLastPromotion INT,
                YearsWithCurrManager INT
            )
        """)
        
        mysql.connection.commit()
        cur.close()
        print("Database tables created/verified successfully")
        
    except Exception as e:
        print(f"Database initialization error: {e}")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)