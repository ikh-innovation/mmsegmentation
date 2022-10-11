import plotly.graph_objects as go


def radar_metrics(metrics: [[]], show=False):

    categories = ['Precision', 'F1 score', 'Recall', 'IoU', 'Accuracy']

    fig = go.Figure()

    for i, m in enumerate(metrics):

        fig.add_trace(go.Scatterpolar(
            r=m,
            theta=categories,
            fill='toself',
            connectgaps=True,
            name='model ' + str(i+1)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )

    if show:
        fig.show()
    else:
        fig.write_image("fig.png")


if __name__ == '__main__':
    radar_metrics([[0.6292,  0.6634, 0.7247, 0.5924, 0.9834], [0.7519, 0.7594 , 0.7671, 0.6479, 0.99]], show=False)
    # radar_metrics([[0.2634, 0.3353, 0.4613, 0.2014, 0.4613], [ 0.5572, 0.5662, 0.5754, 0.3949, 0.5754]], show=False)

