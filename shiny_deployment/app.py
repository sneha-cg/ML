from shiny import App, ui, render
from shinywidgets import output_widget, render_widget
from kmean import plot_elbow, pull_dataset, train_kmeans, get_coef
    

app_ui = ui.page_fluid(
    ui.h1("K-Means Clustering"),
    ui.p("K Mean is a unsupervised algorithm used for clustering. This notebook is used to study K-Mean clustering algorithm by implementing it to cluster the iris dataset according to the main features that is Sepal Length, Sepal Width, Petal Length and Petal Width.", style = "max-width:1000px"),
    ui.h2("About the dataset from origin site"),
    ui.p("This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width."),
    ui.h2("Introducing dataset"),
    ui.output_table("dataset"),
    ui.h2("Choosing the Appropriate Number of Clusters"),
    ui.p("Now we need to find out the appropriate no of clusters to be formed.For that we need to use either Elbow method or silhouette coefficient method."),
    ui.h2("Elbow method"),
    ui.p("The elbow method serves as a visual technique to identify the appropriate value of K in a K-Mean clustering algorithm. The elbow graph illustrates within-cluster-sum-of-square (WCSS) values on the y-axis in relation to various K values on the x-axis. The optimal K value is decided at the point where the graph creates an elbow."),
    ui.h2("Elbow Graph"),
    # plots
    ui.row(
        ui.output_plot("elbow_plot")      
    ),
    ui.p("From the elbow graph we can see that the optimal number of clusters is 2."),
    ui.h2("K-Means Clustering"),
    ui.row(
        ui.output_plot("kmeans_plot")
    ),
    ui.h2("Silhouette coefficient"),
    ui.p("This is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1."),
    ui.h2("Silhouette Coefficient for our model:"),
    ui.output_text('coef')
)

def server(input, output, session):
    @render.table
    def dataset():
        return pull_dataset()
    @render.plot() 
    def elbow_plot():
        return plot_elbow()
    @render.plot() 
    def kmeans_plot():
        return train_kmeans()
    @render.text()
    def coef():
        return get_coef()
    

    
app = App(app_ui, server)
