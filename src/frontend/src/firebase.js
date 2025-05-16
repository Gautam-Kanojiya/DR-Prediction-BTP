// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
const firebaseConfig = {
  apiKey: "AIzaSyBdrLxI4bHYf8XN7yYO4GKcvkWx59XM_NE",
  authDomain: "diabetic-retino-5a276.firebaseapp.com",
  projectId: "diabetic-retino-5a276",
  storageBucket: "diabetic-retino-5a276.appspot.com",
  messagingSenderId: "542366991427",
  appId: "1:542366991427:web:7a226b22664867397a00f6"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

const auth = getAuth();

export { app, auth };