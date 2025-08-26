import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
@st.cache_data
def load_data(filepath):
    """Loads the customer dataset from a CSV file."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please make sure it's in the correct directory.")
        return None

def preprocess_data(df):
    """Selects relevant features and scales them."""
    # For this example, we focus on Annual Income and Spending Score
    features = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, features

# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("📊 Customer Segmentation")
        st.write("""
        This dashboard helps you understand customer behavior by grouping them into segments using clustering algorithms.
        """)
        
        st.subheader("Settings")
        algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN"])

        if algorithm == "K-Means":
            k_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5, step=1,
                                   help="Select the number of customer segments to create.")
        else: # DBSCAN settings
            eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                            help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
            min_samples = st.slider("Minimum Samples", min_value=1, max_value=10, value=5, step=1,
                                    help="The number of samples in a neighborhood for a point to be considered as a core point.")
    
    # --- Load and Preprocess Data ---
    data = load_data('customer_data.csv')
    
    if data is not None:
        scaled_data, original_features = preprocess_data(data)

        # --- Clustering ---
        if algorithm == "K-Means":
            model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            clusters = model.fit_predict(scaled_data)
        else: # DBSCAN
            model = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = model.fit_predict(scaled_data)
            
        data['Cluster'] = clusters

        # --- Main Panel ---
        st.title("Customer Segmentation Analysis")
        st.markdown("---")

        # --- Data Visualization ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customer Segments")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                data=data,
                x='Annual Income (k$)',
                y='Spending Score (1-100)',
                hue='Cluster',
                palette='viridis',
                s=100,
                ax=ax,
                style='Cluster' if 'Cluster' in data.columns and data['Cluster'].nunique() < 10 else None
            )
            ax.set_title(f'Customer Segments using {algorithm}')
            ax.set_xlabel('Annual Income (k$)')
            ax.set_ylabel('Spending Score (1-100)')
            ax.legend(title='Cluster')
            ax.grid(True)
            st.pyplot(fig)

        with col2:
            st.subheader("Principal Component Analysis (PCA)")
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(data=pca_data, columns=['PC 1', 'PC 2'])
            pca_df['Cluster'] = clusters

            fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                data=pca_df,
                x='PC 1',
                y='PC 2',
                hue='Cluster',
                palette='viridis',
                s=100,
                ax=ax_pca,
                style='Cluster' if 'Cluster' in pca_df.columns and pca_df['Cluster'].nunique() < 10 else None
            )
            ax_pca.set_title('PCA of Customer Data')
            ax_pca.set_xlabel('Principal Component 1')
            ax_pca.set_ylabel('Principal Component 2')
            ax_pca.legend(title='Cluster')
            ax_pca.grid(True)
            st.pyplot(fig_pca)

        # --- Data Display ---
        st.markdown("---")
        st.subheader("Customer Data with Cluster Labels")
        st.dataframe(data)

if __name__ == '__main__':
    main()