# Main script to run Iris analysis
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.xkcd()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.graph_objects as go
import umap
from jinja2 import Environment, FileSystemLoader

# ...existing code...

def ensure_dirs(base):
    for d in ['data','outputs','outputs/figs']:
        os.makedirs(os.path.join(base,d), exist_ok=True)


def load_and_split():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test, feature_names, target_names


def save_csvs(base, X_train, X_test, y_train, y_test):
    X_train.to_csv(os.path.join(base,'data','X_train.csv'), index=False)
    X_test.to_csv(os.path.join(base,'data','X_test.csv'), index=False)
    pd.Series(y_train).to_csv(os.path.join(base,'data','y_train.csv'), index=False, header=['target'])
    pd.Series(y_test).to_csv(os.path.join(base,'data','y_test.csv'), index=False, header=['target'])


def plot_pairwise(X, y, target_names, outpath):
    df = X.copy()
    df['target'] = y.values
    df['species'] = df['target'].map(dict(enumerate(target_names)))
    pd.plotting.scatter_matrix(df.drop(columns=['target']), figsize=(10,10), diagonal='kde')
    plt.suptitle('Pairwise feature relationships (XKCD style)')
    plt.savefig(outpath)
    plt.close()


def run_umap(X_train, X_test, n_neighbors=15, min_dist=0.1, random_state=42):
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding_train = reducer.fit_transform(X_train)
    embedding_test = reducer.transform(X_test)
    return embedding_train, embedding_test


def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def make_3d_plot(emb_train, emb_test, y_train, y_test, y_pred, target_names, outpath_html):
    # colors
    colors = ['red','green','blue']
    fig = go.Figure()
    # training points (actual)
    for i, name in enumerate(target_names):
        idx = (y_train==i)
        fig.add_trace(go.Scatter3d(x=emb_train[idx,0], y=emb_train[idx,1], z=emb_train[idx,2], mode='markers',
                                   marker=dict(color=colors[i], size=5), name=f'Train - {name} (actual)'))
    # test points actual
    for i, name in enumerate(target_names):
        idx = (y_test==i)
        fig.add_trace(go.Scatter3d(x=emb_test[idx,0], y=emb_test[idx,1], z=emb_test[idx,2], mode='markers',
                                   marker=dict(color=colors[i], size=8, symbol='diamond'), name=f'Test - {name} (actual)'))
    # test points predicted (overlay with symbol)
    for i, name in enumerate(target_names):
        idx = (y_pred==i)
        fig.add_trace(go.Scatter3d(x=emb_test[idx,0], y=emb_test[idx,1], z=emb_test[idx,2], mode='markers',
                                   marker=dict(color=colors[i], size=6, symbol='cross'), name=f'Test - {name} (predicted)'))
    fig.update_layout(title='3D UMAP embedding: training vs test (actual vs predicted)')
    fig.write_html(outpath_html)


def make_report(context, template_path, outpath):
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    tpl = env.get_template(os.path.basename(template_path))
    html = tpl.render(context)
    with open(outpath,'w', encoding='utf-8') as f:
        f.write(html)


def main():
    base = os.path.dirname(os.path.dirname(__file__))
    ensure_dirs(base)
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_split()
    save_csvs(base, X_train, X_test, y_train, y_test)
    plot_pairwise(X_train, y_train, target_names, os.path.join(base,'outputs','figs','pairwise.png'))
    emb_train, emb_test = run_umap(X_train, X_test)
    clf = train_model(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    make_3d_plot(emb_train, emb_test, y_train.values, y_test.values, y_pred, target_names, os.path.join(base,'outputs','3d_umap.html'))
    # build report
    context = {
        'objectives': 'Analyze Iris dataset, present EDA, model results, and interactive 3D visualization.',
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'fig_pairwise': 'outputs/figs/pairwise.png',
        'fig_3d': 'outputs/3d_umap.html'
    }
    make_report(context, os.path.join(base,'src','report_template.html'), os.path.join(base,'outputs','report.html'))
    print('Report generated at outputs/report.html')

if __name__ == '__main__':
    main()
