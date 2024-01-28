const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const { MongoClient } = require('mongodb');
 
const app = express();
const port = 3000;

// Set up Multer for handling file uploads
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Load MobileNet model
let model;
mobilenet.load().then((loadedModel) => {
  model = loadedModel;
  console.log('MobileNet model loaded');
});

// Connect to MongoDB
const mongoClient = new MongoClient('mongodb+srv://priyanshugupta112002:Samsung@cluster0.9l35amq.mongodb.net/E-commerce', { useUnifiedTopology: true });

async function connectToMongo() {
  try {
    await mongoClient.connect();
    console.log('Connected to MongoDB');
  } catch (error) {
    console.error('Error connecting to MongoDB:', error);
  }
}

// Function to calculate cosine similarity
function cosineSimilarity(a, b) {
  const magnitudeA = a.norm();
  const magnitudeB = b.norm();
  return a.dot(b).div(magnitudeA.mul(magnitudeB));
}

// Process uploaded image and find similar images
async function findSimilarImages(referenceFeatures, databaseImages) {
  // Extract features from each image in the database
  const databaseFeatures = await Promise.all(databaseImages.map(img => model.infer(img)));

  // Calculate cosine similarity between the reference image and database images
  const similarities = databaseFeatures.map(feature => cosineSimilarity(referenceFeatures, feature));

  // Sort the database images based on similarity
  const sortedIndexes = similarities.map((similarity, index) => ({index, similarity}))
                                     .sort((a, b) => b.similarity - a.similarity)
                                     .map(obj => obj.index);

  // Return a list of similar images and their similarity scores
  const similarImages = sortedIndexes.map(i => ({ image: databaseImages[i], similarity: similarities[i] }));

  return similarImages;
}