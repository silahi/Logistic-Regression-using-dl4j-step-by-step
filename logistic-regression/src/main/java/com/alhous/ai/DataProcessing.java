package com.alhous.ai;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;

public class DataProcessing {

    public static void main(String[] args) throws Exception {

        File fIn = new File("C:/deeplearning4j-tutorials/data/iris.data"); 
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(fIn));
        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext()){
            originalData.add(rr.next());
        }
        rr.close();
        
         Schema schema = new Schema.Builder()
                .addColumnsFloat("Sepal Length","Sepal Width", "Petal Length", "Petal Width")
                .addColumnCategorical("Species", Arrays.asList("Iris-setosa","Iris-versicolor","Iris-virginica"))
                .build();
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .categoricalToInteger("Species")
                .build();        
        List<List<Writable>> finalData = LocalTransformExecutor
                .execute(originalData, transformProcess);
        
        RecordWriter rw = new CSVRecordWriter();
        Partitioner partitioner = new NumberOfRecordsPartitioner();
        File fout = new File("C:/deeplearning4j-tutorials/data/iris-pro.csv");
        rw.initialize(new FileSplit(fout), partitioner);
        rw.writeBatch(finalData);
        rw.close();     

    }
}
