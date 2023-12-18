# Import necessary libraries
import pandas as pd
from sklearn import datasets
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the Iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Discretize numerical features into bins
iris_df['sepal length (cm)'] = pd.cut(iris_df['sepal length (cm)'], bins=3, labels=['short', 'medium', 'long'])
iris_df['sepal width (cm)'] = pd.cut(iris_df['sepal width (cm)'], bins=3, labels=['narrow', 'medium', 'wide'])
iris_df['petal length (cm)'] = pd.cut(iris_df['petal length (cm)'], bins=3, labels=['short', 'medium', 'long'])
iris_df['petal width (cm)'] = pd.cut(iris_df['petal width (cm)'], bins=3, labels=['narrow', 'medium', 'wide'])

# Convert the dataset into a one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit_transform(iris_df.values.astype(str))
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Generate association rules
association_rules_result = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Print the frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Print the association rules
print("\nAssociation Rules:")
print(association_rules_result)
# Import necessary libraries for visualization
import matplotlib.pyplot as plt
import networkx as nx

# Plotting frequent itemsets
plt.figure(figsize=(10, 6))
plt.barh(frequent_itemsets['itemsets'].apply(lambda x: ', '.join(x)), frequent_itemsets['support'], color='skyblue')
plt.xlabel('Support')
plt.title('Frequent Itemsets')
plt.show()

# Plotting association rules
plt.figure(figsize=(12, 8))
G = nx.DiGraph()

# Add nodes
for index, row in association_rules_result.iterrows():
    G.add_node(row['antecedents'], color='skyblue', node_size=500)
    G.add_node(row['consequents'], color='salmon', node_size=500)
    G.add_edge(row['antecedents'], row['consequents'], weight=row['confidence'], color='gray')

# Draw graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_color=[G.nodes[node]['color'] for node in G.nodes])
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title('Association Rules Graph')
plt.show()

