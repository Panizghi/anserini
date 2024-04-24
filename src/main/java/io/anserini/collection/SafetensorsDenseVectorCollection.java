/*
 * Anserini: A Lucene toolkit for reproducible information retrieval research
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 package io.anserini.collection;

 import org.python.util.PythonInterpreter;
 import org.python.core.*;
 
 import java.io.BufferedReader;
 import java.io.IOException;
 import java.nio.file.Path;
 import java.util.HashMap;
 import java.util.Map;
 
 /**
  * A document collection for encoded dense vectors for ANN (HNSW) search using Jython.
  * This class uses Jython to execute a Python script that reads and processes SafeTensors data.
  */
 public class SafetensorsDenseVectorCollection extends DocumentCollection<SafetensorsDenseVectorCollection.Document> {
     private Path path;
     private PythonInterpreter interpreter;
 
     public SafetensorsDenseVectorCollection(Path path) {
         this.path = path;
         this.interpreter = new PythonInterpreter();
     }
 
     @Override
     public FileSegment<SafetensorsDenseVectorCollection.Document> createFileSegment(Path p) throws IOException {
         try {
             interpreter.execfile("path/to/your_python_script.py");
             interpreter.set("tensor_file_path", p.toString());
             PyObject tensorData = interpreter.eval("read_tensor_data(tensor_file_path)");
             return new Segment(tensorData);
         } finally {
             interpreter.close();  // Ensure the interpreter is closed after use
         }
     }
 
     public static class Segment extends FileSegment<SafetensorsDenseVectorCollection.Document> {
         private PyObject tensorData;
 
         public Segment(PyObject tensorData) {
             this.tensorData = tensorData;
         }
 
         @Override
         protected Document createNewDocument() {
             PyObject pyObject = tensorData.__getitem__(0); // Get the first item
             String docid = pyObject.__getitem__("docid").toString();
             String contents = pyObject.__getitem__("vector").toString();
 
             return new Document(docid, contents);
         }
     }
 
     public static class Document extends SourceDocument {
         private final String id;
         private final String contents;
 
         public Document(String id, String contents) {
             this.id = id;
             this.contents = contents;
         }
 
         @Override
         public String id() {
             return id;
         }
 
         @Override
         public String contents() {
             return contents;
         }
 
         @Override
         public Map<String, String> fields() {
             return new HashMap<>();
         }
     }
 
     @Override
     public FileSegment<Document> createFileSegment(BufferedReader bufferedReader) throws IOException {
         throw new UnsupportedOperationException("BufferedReader based file segment creation not supported");
     }
 }
 