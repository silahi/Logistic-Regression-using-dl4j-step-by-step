# Logistic regression Using Dl4j step-by-step

In this repository I'm going to demonstrate you how can we use **Deeplearning4j**
for **Logistic Regression** or **Softmax regression** step-by-step. </br> 
We used in this example **Iris Data Set** to classify iris Species. </br> 

These are the key points covered in this example :

### 1. Loading Data using Record Reader and DataSet Iterators
### 2. Transforming Data with Transform Process and Local Transform
### 3. Normalizing Data using Processors
### 4. Partitioning Data into Test Set and Trainning Set
### 5. Configuring a Neural Network and Create the model that uses the configuration
### 6. Evaluate the model and Save the model for future use 

### I. Loading Data

This is an overview of our Data : </br> 
~~~
Sepal Length,Sepal Width,Petal Length, Petal Width,Species
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
...........................
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
............................
6.5,3.0,5.2,2.0,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3.0,5.1,1.8,Iris-virginica
~~~

> Iris Dataset is composed of 150 entries or records. The 1st 50 for the spiece **Iris Setosa**, the 2nd 50 for **Iris Versicolor** and the last 50th for **Iris Virginica**. Each Spiece has a **Sepal Length**, **Sepal Width**, **Petal Length** and **Petal Width**.

To load our data for Transformations and cleaning we use **DataVec Library** :
~~~
File fIn = new File("C:/deeplearning4j-tutorials/data/iris.data"); 
RecordReader rr = new CSVRecordReader(1, ',');
rr.initialize(new FileSplit(fIn));
List<List<Writable>> originalData = new ArrayList<>();
while(rr.hasNext()){
      originalData.add(rr.next());
    }
rr.close();
~~~
> The **CSVRecordReader class** takes in the first param the number of lines to skip in our file **Iris.data**
>And The second param delimiter character : a comma in our case 
> Then we create a **Writable Array** to put the original data Because the data must be in Writable format before any transfromation operation.

### II. Transforming and cleaning 
As our data has no **missing** values nor **nan** values, any cleaning operation is required. But We need to transform categorical values to Integer values.
~~~
Schema schema = new Schema.Builder()
    .addColumnsFloat("Sepal Length","Sepal Width", "Petal Length", "Petal Width")
    .addColumnCategorical("Species", Arrays.asList("Iris-setosa","Iris-versicolor","Iris-virginica"))
    .build();
TransformProcess transformProcess = new TransformProcess.Builder(schema)
    .categoricalToInteger("Species")
    .build();        
List<List<Writable>> finalData = LocalTransformExecutor
    .execute(originalData, transformProcess);
~~~
> Firstly, we create a **schema** to make an in-memory presentation of our data. The categorical column must have in the array the same labels with those in the data.
> And we create a perform a transform process with local Transform execcutor by specifying the Transform process and the schema and the operation (categorical to integer)  

After that we can now save our processed data :
~~~
RecordWriter rw = new CSVRecordWriter();
    Partitioner partitioner = new NumberOfRecordsPartitioner();
    File fout = new File("C:/deeplearning4j-tutorials/data/iris-pro.csv");
    rw.initialize(new FileSplit(fout), partitioner);
    rw.writeBatch(finalData);
    rw.close();
~~~
> we create a record writer and a partitionner then write our data </br>
This is the final result :
~~~
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
.................
7.0,3.2,4.7,1.4,1
6.4,3.2,4.5,1.5,1
6.9,3.1,4.9,1.5,1
.................
6.5,3.0,5.2,2.0,2
6.2,3.4,5.4,2.3,2
5.9,3.0,5.1,1.8,2
~~~
### III. Normalizing Data

So, Before using our data (iris data in our case) for **Learning** or **Training** process, we must before be sure that our data are in the right format, depending of the activation function that will be used. But in our case for logistic regression, we will use **Softmax** that follows a probability distribution. So our data should be in the range **0 to 1** (inclusive) to have better results.</br>
So we are going to load our processed data and then normalize.
~~~
File file = new File("C:/deeplearning4j-tutorials/data/iris-pro.csv");
RecordReader reader = new CSVRecordReader(',');
reader.initialize(new FileSplit(file));
DataSetIterator iterator = new RecordReaderDataSetIterator(reader,150, 4, 3);
        
NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();         preProcessor.fit(iterator);
iterator.setPreProcessor(preProcessor);        
DataSet iriDataSet = iterator.next();

Logger log = LoggerFactory.getLogger(App.class);
log.info("{}",iriDataSet);
~~~
> We load our data via CSVRecordReader, then we precise the record reader, the batch size (150), the number of  features (4) and the number of classes (3) in the dataset iterator. 
> Then we create a Preprocessor, a **NormalizerMinMaxScaler** to normalize our data in range 0 to 1 (inclusive). We can also specify the **min** and the **max** values in the constructor of our normalizer to set values from -1 to 1 for example like this:
~~~
NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler(-1,1); 
~~~
> If you use an activation function like **Hyperbolic Tangent** , it is the best choice but in default the min is **0** and the max is **1**.

When we call the **.next** method of our itarator after setting the preprocessor and fitting the iterator on the preprocessor the data are transformed according to the Normalizer or processor that is used. </br>
This is the resuls when we print the 3rd exampes of our data.

~~~
==================INPUT===================
[[    0.2222,    0.6250,    0.0678,    0.0417], 
 [    0.1667,    0.4167,    0.0678,    0.0417], 
 [    0.1111,    0.5000,    0.0508,    0.0417],
 .............................................
 =================OUTPUT==================
[[    1.0000,         0,         0],
 [    1.0000,         0,         0],
 [    1.0000,         0,         0],
...............................................
~~~
Notice that the output that contained 1 column has now 3 columns witch coresponds to the number of classes because our network will contains 3 neurones in the output layer.


### VI. Data Partition

Now we have our dataset, we split it in two two parts: one part for training and the other part for testing.
~~~
DataSet iriDataSet = iterator.next();
iriDataSet.shuffle(50); 
SplitTestAndTrain testAndTrain = iriDataSet.splitTestAndTrain(0.8);
DataSet trainSet = testAndTrain.getTrain();
DataSet testSet = testAndTrain.getTest();
~~~
> We have shufled or data 50 times then we divide them in to test data and train data. We use **80%** for training and **20%** for testing.

### V. Neural Network configiration

In logistic regression, we need 2 layers : </br>
* **An input layer** that contains the features
* **An output layer** for the probabily computation

So the input layer is not concidered as a layer because it just put the data into the network. We create though the output layer for logistic regression witch is the simplest neural network with one layer.

~~~
OutputLayer outputLayer = new OutputLayer.Builder()
        .nIn(4).nOut(3)            
        .weightInit(WeightInit.XAVIER)  
        .activation(Activation.SOFTMAX)            
        .build();
~~~
> Firstly, we specify the number of input neurons (number of feature) : **4 features**
> Secondly, we specify the number of output neurons (number of classes) : **3 classes**
> Thirdly, we specify the weight initalization algorithm (XAVIER)
> And finaly, the activation function : (SOFTMAX) for logistic regression

The network configuration : 
~~~
MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
        .seed(123)
        .weightInit(WeightInit.XAVIER)
        .updater(new Nesterovs(0.1, 0.9))
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .list()
        .layer(outputLayer)
        .build();

MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
~~~
> We have created our network configuration and specifying the updater algorithm : **Nesterovs** with **learning rate** (0.1) and the **momentum** (0.9) and as optimization algorithm **STOCHASTIC GRADIENT DESCENT**. Then we create the network and initilize the configuration.

### VI. Learning Process
Now we have all we need to train our model by just calling the **fit** method and passing the data.

~~~ 
for (int i =0; i< 1000; i++){
     model.fit(trainSet); 
}
~~~
 
### VII. Model Evaluation

After learning we need to evaluate our model by using the test dataset.
~~~
List<DataSet> list = testSet.asList();    
DataSetIterator testIterator = new ListDataSetIterator<>(list); 
Evaluation eval = model.evaluate(testIterator);
log.info("Precision : {}",eval.precision()); 
log.info("Recall : {}",eval.recall()); 
log.info("Accuracy : {}", eval.accuracy()) ;  
log.info("\n-----Confusion matrix-----\n") ;
log.info("{}",eval.confusionMatrix());
~~~
> After 1000 epochs i got an accuracy of : **0.9333333333333333**
> And this is the confusion matrix </br>
~~~
  0  1  2
----------
 12  0  0 | 0 = 0
  0  7  1 | 1 = 1
  0  1  9 | 2 = 2
~~~

**The complete code is present in the logistic regression folder**