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

 package io.anserini.index.generator;

 import io.anserini.collection.SourceDocument;
 import io.anserini.index.Constants;
 import org.apache.lucene.document.BinaryDocValuesField;
 import org.apache.lucene.document.Document;
 import org.apache.lucene.document.Field;
 import org.apache.lucene.document.KnnFloatVectorField;
 import org.apache.lucene.document.StringField;
 import org.apache.lucene.index.VectorSimilarityFunction;
 import org.apache.lucene.util.BytesRef;
 
 import java.nio.ByteBuffer;
 import java.nio.FloatBuffer;
 import java.nio.file.Files;
 import java.nio.file.Paths;
 
 /**
  * Converts a {@link SourceDocument} into a Lucene {@link Document}, ready to be indexed,
  * using dense vector data retrieved from SafeTensor files.
  *
  * @param <T> type of the source document
  */
 public class HnswSafetensorsDenseVectorDocumentGenerator<T extends SourceDocument> implements LuceneDocumentGenerator<T> {
   
   public HnswSafetensorsDenseVectorDocumentGenerator() {
   }
 
   private float[] readTensor(String tensorPath) throws Exception {
     byte[] tensorBytes = Files.readAllBytes(Paths.get(tensorPath));
     ByteBuffer byteBuffer = ByteBuffer.wrap(tensorBytes);
     FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
 
     float[] vector = new float[floatBuffer.remaining()];
     floatBuffer.get(vector);
     return vector;
   }
 
   @Override
   public Document createDocument(T src) throws InvalidDocumentException {
     String id = src.id();
     float[] contents;
 
     try {
       contents = readTensor(src.contents());  // Assume src.contents() returns the path to the tensor file
     } catch (Exception e) {
       throw new InvalidDocumentException("Failed to read vector data: " + e.getMessage());
     }
 
     // Make a new, empty document.
     final Document document = new Document();
 
     // Store the collection docid.
     document.add(new StringField(Constants.ID, id, Field.Store.YES));
     // This is needed to break score ties by docid.
     document.add(new BinaryDocValuesField(Constants.ID, new BytesRef(id)));
 
     document.add(new KnnFloatVectorField(Constants.VECTOR, contents, VectorSimilarityFunction.DOT_PRODUCT));
 
     return document;
   }
 }
 