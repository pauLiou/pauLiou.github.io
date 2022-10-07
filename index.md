---
layout: default
---

**Hypothesis Machine**

_I have developed an R Shiny application that allows users to perform statistical analysis on two groups and gives high quality presentable outputs_

<iframe height="650" width="100%" frameborder="no" src="https://paulfisher.shinyapps.io/r_shiny_app/">
</iframe>


We simply add a file that has two columns and headers and we can instantly perform statistical calculations. Some sample data has been provided in order to 
test it. Feel free to click around and try it out, you can also upload your own csv file if you like! It just needs 2 columns and a header.

# Step 1: Developing the R code

Here is a brief description of how R Shiny operates:

<p align="center">
<img src="http://yuml.me/diagram/scruffy/class/[Input Data]->[UI Function]-Reactively Cycles>[Server Function]->[UI Function]<>->[Processing/Analysis]<>->[Output Result]" >
</p>
The UI and Server functions communicate back and forth sending input and output information. We begin with the UI function:

## UI function

```R
// Here is R code from my script
ui <- fluidPage(
  tags$style(type='text/css', ".selectize-dropdown { font-size: 12px; line-height:12px; }"),
  theme = shinytheme("superhero"),
  # App Title
  titlePanel("Hypothesis Machine!"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    # Sidebar panel for inputs ----
    sidebarPanel(
      helpText(strong("Upload your csv file here:")), #  for uploading the csv file
      fileInput(
        "file1", # variable name
        "Choose CSV File",
        multiple = TRUE,
        accept = c(".csv") # types accepted
      ),
      tags$hr(),
      helpText(strong("Or manually enter your data below")),
      br(), # page break
      br(),

      ...
     
  mainPanel(plotOutput("contents"),
            textOutput("resultText"))
))
      
```

The UI function deals with inputs in the **sidebarPanel** call. Within this section we can adapt what we will see when the app is initially loaded.
So for example, currently we are only seeing the text instructing us to upload a csv file. Along with the tests that we can perform and a zoom function.
So the first step is to input some data.

The **mainPanel** call is what is output after the inputs are sent to the server function.

## Server function

```R
server <-  function(input, output) {
  output$contents <- renderPlot({
    req(input$file1, file.exists(input$file1$datapath)) # check that the files exist
    req(input$slider1) # also confirm that the slider is working as intended
    x <- read.csv(input$file1$datapath) # read the csv file
    data <- melt(x)
    p <- as.character(unique(data$variable)) # find the unique string names of the columns
    x <- as_tibble(x) # make the data cleaner
    x1 <- ls(x)[[1]] # find the variable name of x1
    x2 <- ls(x)[[2]] # find the variable name of x2
    
    if (input$select == 1) {
     
      fig1 <- ggscatter(data=x,x=x1,y=x2,color = factor(x1),
                add = "reg.line",
                conf.int = TRUE,
                palette = "jco",
                cor.coef = TRUE,
                cor.method = methodR,
                cor.coeff.args = list(label.y = max(x[2]) + max(x[2])/100*5),
                title = "Scatter plot of the two variables with regression line") +
        theme(legend.position = "none")#  scatter plot with regression line
      
      d <- dist(scale(x),method="euclidean",diag=TRUE,upper=TRUE) # creating a hierarchical clustering model
      hls <- hclust(d,method="complete") # compiling the clusters
      
      cluster <- cutree(hls,4) # distributing to 4 bins
      
      ggData <- cbind(x,cluster) # combining the data
      ggData$cluster <- as.factor(ggData$cluster)
      borderX <- max(x[1])/100*input$slider3
      borderY <- max(x[2])/100*input$slider3
      
      fig2 <- ggplot(ggData, aes_string(x=(p[1]),y=(p[2]),color=factor(cluster)))+
        geom_point(size=2.3) +
        ylim(min(x[2]) - borderY, max(x[2]) + borderY) +
        xlim(min(x[1]) - borderX, max(x[1]) + borderX) +
      geom_mark_circle(aes(color = factor(cluster),fill = cluster)) +
        ggtitle("Hierarchical clustering model") +
        theme_classic() +
        theme(legend.position = "none")

      figure <- grid.arrange(
        fig1,
        fig2,
        ncol = 2)
        
      etc...
```
    
The Server function is where we perform calculations and develop our figures. The basic way that I've approached this is to use conditional statements.
If the user selects correlation they will then change the output for the **mainPanel**. So we are conditionally changing the contents of output depending on inputs.
This way people can adjust their selection on the fly and it will still give the desired result. The R Shiny package is reactive programming in it's design. So the **UI**
and **Server** functions run asynchronously to enable interactive feedback.

* * *

# Going into the statistics

## Correlation

```R
if (input$select == 1) {

      
      if(input$method2 == 1){
        
        res <- cor.test(as.numeric(unlist(x[1])),as.numeric(unlist(x[2])),
                        method = "pearson")# run the correlation, as.numeric(unlist) is how we are converting the data type to array
      } else if(input$method2 == 2){
        
        res <- cor.test(as.numeric(unlist(x[1])),as.numeric(unlist(x[2])),
                        method = "spearman")
      } else if(input$method2 == 3){
        
        res <- cor.test(as.numeric(unlist(x[1])),as.numeric(unlist(x[2])),
                        method = "kendall")
      }
     
      if(res$estimate >= 0){
        corDirection = "positive"
      } else{
        corDirection = "negative"
      }
```

Here is a snippet of the code for the correlation analysis. When we calculate a correlation, we are trying to determine the relationship between
two variables. In the provided example, that is age*earnings. When we want to look at a distribution like this. First we need to test if the data
is **parametric** or not. This means that we have to run a test of normal distribution. So the first thing we should do, is jump across onto the **normal distribution**
tab and check the results of this test. If it tells us that it is normally distributed. Then we can run a parametric correlation, which in this case 
is the **Pearson** correlation.

Here is the Pearson's correlation equation:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?r = \frac{\sum{(x-m_x)(y-m_y)}}{\sqrt{\sum{(x-m_x)^2}\sum{(y-m_y)^2}}} {} t " /> 
</p>

Where _m_<sub>_x_</sub> and _m_<sub>_y_</sub> are the means of your two variables
**x** and **y** are the two vectors with a length of **n**

Afterwhich we can perform a simple calculation to determine the **p-value** by calculating _t_.

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?t = \frac{r}{\sqrt{1-r^2}}\sqrt{n-2} {} t " /> 
</p>


We can then look up the result on the following t-distribution table:

|--------|-------|-------|-------|-------|-------|--------|--------|-------|--------|
| df/α   | 0.9   | 0.5   | 0.3   | 0.2   | 0.1   | 0.05   | 0.02   | 0.01  | 0.001  |
|        |       |       |       |       |       |        |        |       |        |
| 1      | 0.158 | 1     | 2     | 3.078 | 6.314 | 12.706 | 31.821 | 64    | 637    |
| 2      | 0.142 | 0.816 | 1.386 | 1.886 | 2.92  | 4.303  | 6.965  | 10    | 31.598 |
| 3      | 0.137 | 0.765 | 1.25  | 1.638 | 2.353 | 3.182  | 4.541  | 5.841 | 12.929 |
| 4      | 0.134 | 0.741 | 1.19  | 1.533 | 2.132 | 2.776  | 3.747  | 4.604 | 8.61   |
| 5      | 0.132 | 0.727 | 1.156 | 1.476 | 2.015 | 2.571  | 3.365  | 4.032 | 6.869  |
|        |       |       |       |       |       |        |        |       |        |
| 6      | 0.131 | 0.718 | 1.134 | 1.44  | 1.943 | 2.447  | 3.143  | 3.707 | 5.959  |
| 7      | 0.13  | 0.711 | 1.119 | 1.415 | 1.895 | 2.365  | 2.998  | 3.499 | 5.408  |
| 8      | 0.13  | 0.706 | 1.108 | 1.397 | 1.86  | 2.306  | 2.896  | 3.355 | 5.041  |
| 9      | 0.129 | 0.703 | 1.1   | 1.383 | 1.833 | 2.263  | 2.821  | 3.25  | 4.781  |
| 10     | 0.129 | 0.7   | 1.093 | 1.372 | 1.812 | 2.228  | 2.764  | 3.169 | 4.587  |
|        |       |       |       |       |       |        |        |       |        |
| Infini | 0.126 | 0.674 | 1.036 | 1.282 | 1.645 | 1.96   | 2.326  | 2.576 | 3.291  |

We can use this look-up table to find our significance value.

[Click here to see full table](http://www.sthda.com/english/wiki/t-distribution-table)

If our data is not normally distributed we can compute the same calculations using either the
**Spearman correlation formula** or the **Kendall correlation formula**. But we won't go into 
details about those calculations here. Either way, the Shiny App takes care of all that for you!

## Paired Samples _t_ test

If we want to compare the mean values of two sets of variables and test whether they are different
to a strong degree. Say, typically 95% certain that they are different. Then we can perform a simple
paired samples _t_ test.

In the provided example we are looking at the difference between **Test Scores** in the **day** and in the **night**. We have 200
data points for each group. Firstly we need to make sure we are following certain assumptions. 

For a _paired_ samples _t_ test. The data needs to have come from the same group. So these test results are collected from the same 
people who completed a test during the day and also during the night.

### We can then compute the following hypotheses:

H<sub>0</sub>: µ<sub>1</sub> = µ<sub>2</sub> _**The paired population means are equal**
H<sub>1</sub>: µ<sub>1</sub> = µ<sub>2</sub> **The paired population means are _not_ equal**

Where µ<sub>1</sub> and µ<sub>2</sub> are the population means of the two samples.

So we therefore have two hypotheses: **H<sub>0</sub>** and **H<sub>1</sub>**.

The goal of the _t_ test is to determine if our data fits into the **H<sub>0</sub> or _NULL_ hypothesis assumption, or is statistically
different (significantly different) to the null hypothesis.

A _t_ test operates in the following manner:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?t = \frac{m}{s/\sqrt{n}}" /> 
</p>

**_m_** = the mean difference between each pair of values.

**_n_** = the size of the sample

**_s_** = the standard deviation of

## Normal Distribution

The normal distribution is important for determining whether our data is **parametric** or **non-parametric**. We can do this by plotting
the density/frequency of the data in a histogram. We are mostly concerned with what's known as the **Central Limit Theorem**. Simply put we want
to determine whether our data is nicely collected around our mean score without too many values skewing the data to one side and without too many
outliers. The majority of our data should nicely sit within **one standard deviation** from the mean if we want it to be considered normally distributed.

Luckily for us, there are some tools we can use to determine this.

Firstly we can plot the density and histogram of the data (demonstrated in the Shiny App).

We can also perform what's known as a **Q-Q plot** or **Quantile-Quantile plot** which gives us the correlation between a given sample and the normal
distribution with a 45-degree reference.

### The Shapiro-Wilk's test

Unfortunately for us, whilst visualising the data in this way is helpful and important. We cannot rely on this fully to determine whether our data is normally 
distributed. An eye-test is just not sufficiently accurate. One such tool that can handle this for us is the **Shapiro-Wilk's test of normality**.

This test is give us a much more reliable indication of whether or not our data is normally distributed.

It's also incredibly simple to perform in R:

```R
normality <- shapiro.test(data$value) # the shapiro test for normal distribution
```

And that's that! You can also look at the descriptive statistics of the data which probably doesn't need much describing.

The code for my Hypothesis Machine is available on my github main page [here](https://github.com/pauLiou/R-Portfolio/)