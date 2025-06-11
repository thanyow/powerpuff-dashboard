
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Loyalty Prediction Dashboard",
    page_icon="🎯",
)
st.title("Customer Loyalty Prediction App")
st.write("Enter customer data to predict whether the customer is loyal or not.")

st.markdown("---")
st.subheader("Model Evaluation")

# Load data and model
@st.cache_data
def load_data_and_model():
    data = pd.read_csv("Customer Purchasing Behaviors.csv")
    model = joblib.load("model.pkl")
    return data, model

data, model = load_data_and_model()

# Prepare data
data = data.drop(['user_id'], axis=1)

region_map = {'North':0, 'South':1, 'West':2, 'East':3}
data['region'] = data['region'].map(region_map)

data['loyal_customer'] = (data['loyalty_score'] >= 7.0).astype(int)

X = data.drop(['loyalty_score', 'loyal_customer'], axis=1)
y = data['loyal_customer']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# If model is not fitted, fit it (optional)
try:
    model.n_estimators
except AttributeError:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

# Predict all data for evaluation
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Calculate metrics
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)

col1, col2, col3, col4 = st.columns(4)
col1.success(f"Accuracy: **{acc:.2f}**")
col2.info(f"Precision: **{prec:.2f}**")
col3.warning(f"Recall: **{rec:.2f}**")
col4.error(f"ROC AUC: **{roc_auc:.2f}**")

plot_option = st.selectbox("Select the graph to display:", ["Select", "ROC AUC Curve", "Confusion Matrix", "Customer Clusters", "Cluster Analysis"])

if plot_option == "ROC AUC Curve":
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)

elif plot_option == "Confusion Matrix":
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

elif plot_option == "Customer Clusters":
    st.subheader("Customer Clustering Analysis")
    
    # Prepare data for clustering
    cluster_features = ['age', 'annual_income', 'purchase_amount', 'purchase_frequency']
    X_cluster = data[cluster_features].copy()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Perform K-means clustering
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=3, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    data_with_clusters['Cluster'] = data_with_clusters['Cluster'].astype(str)
    
    # 2D scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = px.scatter(
            data_with_clusters,
            x='annual_income',
            y='purchase_amount',
            color='Cluster',
            size='purchase_frequency',
            hover_data=['age', 'loyalty_score'],
            title='Income vs Purchase Amount'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = px.scatter(
            data_with_clusters,
            x='age',
            y='purchase_frequency',
            color='Cluster',
            size='annual_income',
            hover_data=['purchase_amount', 'loyalty_score'],
            title='Age vs Purchase Frequency'
        )
        st.plotly_chart(fig3, use_container_width=True)

elif plot_option == "Cluster Analysis":
    st.subheader("Detailed Cluster Analysis")
    
    # Prepare data for clustering
    cluster_features = ['age', 'annual_income', 'purchase_amount', 'purchase_frequency']
    X_cluster = data[cluster_features].copy()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Perform K-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    # Cluster statistics
    st.write("### Cluster Statistics")
    cluster_stats = data_with_clusters.groupby('Cluster').agg({
        'age': ['mean', 'std'],
        'annual_income': ['mean', 'std'],
        'purchase_amount': ['mean', 'std'],
        'purchase_frequency': ['mean', 'std'],
        'loyalty_score': ['mean', 'std'],
        'loyal_customer': ['mean', 'count']
    }).round(2)
    
    st.dataframe(cluster_stats)
    
    # Loyalty distribution by cluster
    st.write("### Loyalty Distribution by Cluster")
    loyalty_by_cluster = data_with_clusters.groupby(['Cluster', 'loyal_customer']).size().unstack(fill_value=0)
    loyalty_by_cluster['Loyalty_Rate'] = (loyalty_by_cluster[1] / (loyalty_by_cluster[0] + loyalty_by_cluster[1]) * 100).round(2)
    
    fig_bar = px.bar(
        x=loyalty_by_cluster.index,
        y=loyalty_by_cluster['Loyalty_Rate'],
        title='Loyalty Rate by Cluster (%)',
        labels={'x': 'Cluster', 'y': 'Loyalty Rate (%)'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Cluster characteristics
    st.write("### Cluster Characteristics")
    for cluster_id in sorted(data_with_clusters['Cluster'].unique()):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
        loyalty_rate = (cluster_data['loyal_customer'].sum() / len(cluster_data) * 100).round(1)
        
        with st.expander(f"Cluster {cluster_id} - Loyalty Rate: {loyalty_rate}%"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Age", f"{cluster_data['age'].mean():.1f}")
                st.metric("Average Income", f"${cluster_data['annual_income'].mean():,.0f}")
            
            with col2:
                st.metric("Average Purchase Amount", f"${cluster_data['purchase_amount'].mean():.0f}")
                st.metric("Average Purchase Frequency", f"{cluster_data['purchase_frequency'].mean():.1f}")
            
            with col3:
                st.metric("Average Loyalty Score", f"{cluster_data['loyalty_score'].mean():.2f}")
                st.metric("Total Customers", f"{len(cluster_data)}")

st.markdown("---")
st.subheader("Customer Loyalty Prediction")

with st.form("prediction_form"):

    region_input = st.selectbox("Region", ['North', 'South', 'West', 'East'])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    annual_income = st.number_input("Annual Income", min_value=0, value=50000)
    purchase_amount = st.number_input("Purchase Amount", min_value=0, value=200)
    purchase_frequency = st.number_input("Purchase Frequency", min_value=0, value=10)

    submit = st.form_submit_button("Predict Loyalty")

def map_input():
    region_map = {'North':0, 'South':1, 'West':2, 'East':3}
    return pd.DataFrame([{
        'age': age,
        'annual_income': annual_income,
        'purchase_amount': purchase_amount,
        'region': region_map[region_input],
        'purchase_frequency': purchase_frequency
    }])

def predict_cluster(input_data):
    """Predict which cluster the new customer belongs to"""
    cluster_features = ['age', 'annual_income', 'purchase_amount', 'purchase_frequency']
    X_cluster = data[cluster_features].copy()
    
    # Scale the features
    scaler = StandardScaler()
    scaler.fit(X_cluster)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    X_scaled = scaler.transform(X_cluster)
    kmeans.fit(X_scaled)
    
    # Scale and predict cluster for new customer
    input_scaled = scaler.transform(input_data[cluster_features])
    predicted_cluster = kmeans.predict(input_scaled)[0]
    
    return predicted_cluster

if submit:
    try:
        input_df = map_input()

        # Predict loyalty
        proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        
        # Predict cluster
        predicted_cluster = predict_cluster(input_df)

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)
        
        with col1:
            # Show loyalty prediction
            if pred == 1:
                st.success(f"The customer is predicted to be: **LOYAL** 🎉")
                st.write(f"Loyalty probability: **{proba:.2f}**")
            else:
                st.error(f"The customer is predicted to be: **NOT LOYAL** 😞")
                st.write(f"Loyalty probability: **{proba:.2f}**")
        
        with col2:
            st.info(f"**Customer Segment: Cluster {predicted_cluster}**")
            
            # Show cluster characteristics
            cluster_data = data[data.groupby(['age', 'annual_income', 'purchase_amount', 'purchase_frequency']).ngroup() % 4 == predicted_cluster]
            if not cluster_data.empty:
                avg_loyalty_rate = (data.groupby(KMeans(n_clusters=4, random_state=42).fit_predict(
                    StandardScaler().fit_transform(data[['age', 'annual_income', 'purchase_amount', 'purchase_frequency']])
                ))['loyal_customer'].mean()[predicted_cluster] * 100)
                st.write(f"Typical loyalty rate for this segment: **{avg_loyalty_rate:.1f}%**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Make sure the input data matches the expected format.")
