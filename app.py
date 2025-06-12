
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Customer Loyalty Prediction Dashboard", page_icon="🎯")
st.title("🎯 Customer Loyalty Prediction App")
st.write("Enter customer data to predict whether the customer is loyal or not.")

st.markdown("---")
st.subheader("📊 Model Evaluation")

# Load data and models
@st.cache_data
def load_resources():
    try:
        df = pd.read_csv("Customer Purchasing Behaviors.csv")
        model = joblib.load("model.pkl")
        scaler_kmeans = joblib.load("scaler_kmeans.pkl")
        scaler_logreg = joblib.load("scaler_logreg.pkl")
        kmeans = joblib.load("kmeans_model.pkl")

        return df, model, scaler_kmeans, scaler_logreg, kmeans
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

df, model, scaler_kmeans, scaler_logreg, kmeans = load_resources()

# Preprocessing
region_map = {'North': 0, 'South': 1, 'West': 2, 'East': 3}
df = df.drop(['user_id'], axis=1, errors='ignore')
df['region_encoded'] = df['region'].map(region_map)
df['loyal_customer'] = (df['loyalty_score'] >= 7).astype(int)

# Debug: Check what features the scaler and kmeans expect
st.sidebar.write("✅ scaler_kmeans loaded")
st.sidebar.write("✅ scaler_logreg loaded")

# Handle feature mismatch between scaler and kmeans
try:
    # Use only the 3 clustering features used during model training
    cluster_features = ['age', 'annual_income', 'purchase_amount']
    X_cluster = df[cluster_features]
    X_scaled = scaler_kmeans.transform(X_cluster)
    df['Cluster'] = kmeans.predict(X_scaled)
except Exception as e:
    st.error(f"Error in clustering: {e}")
    st.write("Make sure the clustering model and scaler are trained on the same 3 features.")
    st.stop()

    # Alternative: Create clusters without using the saved KMeans model
    try:
        from sklearn.cluster import KMeans
        # Create new KMeans with same number of clusters as the original
        n_clusters = len(np.unique(kmeans.labels_)) if hasattr(kmeans, 'labels_') else 4
        new_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Use available numeric features for clustering
        cluster_cols = ['age', 'annual_income', 'purchase_amount', 'purchase_frequency']
        X_cluster = df[cluster_cols]
        
        # Standardize the features
        from sklearn.preprocessing import StandardScaler
        temp_scaler = StandardScaler()
        X_scaled = scaler_logreg.transform(X)
        
        # Fit and predict
        df['Cluster'] = new_kmeans.fit_predict(X_scaled)
        st.warning("Used alternative clustering due to model mismatch")
        
    except Exception as e2:
        st.error(f"Alternative clustering also failed: {e2}")
        # Create dummy clusters as last resort
        df['Cluster'] = np.random.randint(0, 4, size=len(df))
        st.warning("Using random clusters as fallback")

# Analyze cluster stats
def analyze_clusters():
    results = {}
    for cluster_id in sorted(df['Cluster'].unique()):
        data = df[df['Cluster'] == cluster_id]
        results[cluster_id] = {
            'avg_age': data['age'].mean(),
            'avg_income': data['annual_income'].mean(),
            'avg_purchase': data['purchase_amount'].mean(),
            'avg_freq': data['purchase_frequency'].mean(),
            'avg_loyalty': data['loyalty_score'].mean(),
            'loyalty_rate': (data['loyal_customer'].mean() * 100),
            'size': len(data)
        }
    return results

cluster_stats = analyze_clusters()

# Generate readable names
def generate_cluster_names(stats):
    names = {}
    for cid, val in stats.items():
        age = "Young" if val['avg_age'] < 35 else "Middle-aged" if val['avg_age'] < 50 else "Mature"
        income = "High" if val['avg_income'] > 60000 else "Medium" if val['avg_income'] > 40000 else "Low"
        freq = "High" if val['avg_freq'] > 15 else "Medium" if val['avg_freq'] > 8 else "Low"
        loyalty = "High" if val['loyalty_rate'] > 70 else "Medium" if val['loyalty_rate'] > 30 else "Low"
        names[cid] = f"{age} {income}-Income {freq}-Frequency ({loyalty} Loyalty)"
    return names

cluster_names = generate_cluster_names(cluster_stats)

# Classification prep - use the same features as the model expects
# Assuming the model was trained with these 5 features based on the error
model_features = ['age', 'annual_income', 'purchase_amount', 'purchase_frequency', 'region_encoded']
X = df[model_features]
y = df['loyal_customer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

try:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.success(f"Accuracy: **{acc:.2f}**")
    col2.info(f"Precision: **{prec:.2f}**")
    col3.warning(f"Recall: **{rec:.2f}**")
    col4.error(f"ROC AUC: **{roc_auc:.2f}**")
    
except Exception as e:
    st.error(f"Error in model evaluation: {e}")
    st.write("Model expects features:", model_features)
    acc = prec = rec = f1 = roc_auc = 0  # Default values for display

# Visual Options
st.subheader("📈 Visualize")
plot_option = st.selectbox("Choose a visualization:", ["Select", "ROC AUC Curve", "Confusion Matrix", "Customer Clusters", "Cluster Analysis"])

if plot_option == "ROC AUC Curve" and roc_auc > 0:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

elif plot_option == "Confusion Matrix" and acc > 0:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

elif plot_option == "Customer Clusters":
    st.subheader("🧩 Customer Clustering")
    try:
        fig1 = px.scatter(df, x='annual_income', y='purchase_amount', color='Cluster', 
                          size='purchase_frequency', hover_data=['age', 'loyalty_score'], 
                          title='Income vs Purchase Amount by Cluster')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter(df, x='age', y='purchase_frequency', color='Cluster', 
                          size='annual_income', hover_data=['purchase_amount', 'loyalty_score'], 
                          title='Age vs Purchase Frequency by Cluster')
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating cluster visualizations: {e}")

elif plot_option == "Cluster Analysis":
    st.subheader("🔍 Cluster Analysis")
    try:
        cluster_df = pd.DataFrame([
            {
                "Cluster": cid,
                "Name": cluster_names[cid],
                "Avg Age": f"{val['avg_age']:.1f}",
                "Avg Income": f"${val['avg_income']:,.0f}",
                "Avg Purchase": f"${val['avg_purchase']:.0f}",
                "Avg Frequency": f"{val['avg_freq']:.1f}",
                "Avg Loyalty": f"{val['avg_loyalty']:.2f}",
                "Loyalty Rate": f"{val['loyalty_rate']:.1f}%",
                "Customers": val['size']
            }
            for cid, val in cluster_stats.items()
        ])
        st.dataframe(cluster_df, use_container_width=True)

        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        st.info(f"Silhouette Score for k={kmeans.n_clusters}: **{sil_score:.2f}**")
        st.write("Silhouette Score indicates how well-separated the clusters are. Higher values indicate better-defined clusters.")

    except Exception as e:
        st.error(f"Error creating cluster analysis: {e}")

# --- Prediction Form ---
st.markdown("---")
st.subheader("🧠 Predict Customer Loyalty")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        region = st.selectbox("Region", list(region_map.keys()))
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        income = st.number_input("Annual Income ($)", min_value=0, max_value=200000, value=50000)
    
    with col2:
        amount = st.number_input("Purchase Amount ($)", min_value=0, max_value=10000, value=300)
        frequency = st.number_input("Purchase Frequency", min_value=0, max_value=100, value=10)
    
    submit = st.form_submit_button("🎯 Predict Loyalty", use_container_width=True)

def predict_segment(input_df):
    try:
        X = input_df[['age', 'annual_income', 'purchase_amount']]
        X_scaled = scaler_kmeans.transform(X)
        return kmeans.predict(X_scaled)[0]
    except Exception as e:
        st.error(f"Error in segment prediction: {e}")
        return 0


if submit:
    try:
        # Create input data with all required features
        input_data = pd.DataFrame([{
            "age": age,
            "annual_income": income,
            "purchase_amount": amount,
            "purchase_frequency": frequency,
            "region_encoded": region_map[region]
        }])

        # Predict segment
        cluster_id = predict_segment(input_data)
        segment_name = cluster_names.get(cluster_id, "Unknown Segment")
        
        # Predict loyalty
        loyalty_pred = model.predict(input_data)[0]
        loyalty_prob = model.predict_proba(input_data)[0][1]

        st.subheader("🎯 Prediction Results")
        
        # Create two columns for results
        col1, col2 = st.columns(2)

        with col1:
            label = "LOYAL" if loyalty_pred == 1 else "NOT LOYAL"
            emoji = "🎉" if loyalty_pred == 1 else "😞"
            color = "success" if loyalty_pred == 1 else "error"
            
            st.markdown(f"### Loyalty Prediction")
            if loyalty_pred == 1:
                st.success(f"**{label}** {emoji}")
            else:
                st.error(f"**{label}** {emoji}")
            
            st.metric("Loyalty Probability", f"{loyalty_prob:.1%}")

        with col2:
            st.markdown(f"### Customer Segment")
            st.info(f"**Cluster {cluster_id}**")
            st.write(f"*{segment_name}*")
            st.metric("Segment Loyalty Rate", f"{cluster_stats[cluster_id]['loyalty_rate']:.1f}%")

        # Additional insights
        st.markdown("---")
        st.subheader("📊 Customer Insights")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age Group", 
                     "Young" if age < 35 else "Middle-aged" if age < 50 else "Mature")
        with col2:
            st.metric("Income Level", 
                     "High" if income > 60000 else "Medium" if income > 40000 else "Low")
        with col3:
            st.metric("Purchase Behavior", 
                     "High" if frequency > 15 else "Medium" if frequency > 8 else "Low")

        # Summary for download
        summary = f"""Customer Loyalty Prediction Report
==========================================

PREDICTION RESULTS:
- Loyalty Status: {label} {emoji}
- Confidence: {loyalty_prob:.1%}
- Customer Segment: Cluster {cluster_id}
- Segment Description: {segment_name}
- Segment Loyalty Rate: {cluster_stats[cluster_id]['loyalty_rate']:.1f}%

CUSTOMER PROFILE:
- Age: {age} years
- Annual Income: ${income:,}
- Average Purchase: ${amount:,}
- Purchase Frequency: {frequency} times
- Region: {region}

SEGMENT CHARACTERISTICS:
- Average Age: {cluster_stats[cluster_id]['avg_age']:.1f} years
- Average Income: ${cluster_stats[cluster_id]['avg_income']:,.0f}
- Average Purchase: ${cluster_stats[cluster_id]['avg_purchase']:.0f}
- Average Frequency: {cluster_stats[cluster_id]['avg_freq']:.1f}
- Average Loyalty Score: {cluster_stats[cluster_id]['avg_loyalty']:.2f}
- Total Customers in Segment: {cluster_stats[cluster_id]['size']}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        st.download_button(
            "📥 Download Detailed Report", 
            data=summary, 
            file_name=f"loyalty_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt", 
            mime="text/plain",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Please check that all models are properly trained and saved.")

# Footer
st.markdown("---")
st.markdown("*Powerpuff Girls Group | Customer Loyalty Prediction Dashboard*") 
