const express = require("express");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node"); 
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());

const upload = multer({
  limits: { fileSize: 1000000 }, 
  fileFilter(req, file, cb) {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("File must be an image"));
    }
    cb(null, true);
  },
});

// Load Model
let model;
const modelPath = path.join(__dirname, "models/model.json");
console.log(modelPath);
async function loadModel() {
  try {
    model = await tf.loadGraphModel(`file://${modelPath}`);
    console.log("Model loaded successfully");
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

loadModel();

app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    const file = req.file;

    if (!file) {
      return res.status(400).json({
        status: "fail",
        message: "No file uploaded",
      });
    }

    const buffer = fs.readFileSync(file.path);
    const tensor = tf.node.decodeImage(buffer, 3);
    const resized = tf.image.resizeBilinear(tensor, [224, 224]);
    const normalized = resized.div(255).expandDims(0); 

    const prediction = await model.predict(normalized).dataSync(); // Replace with your model's specific output
    const isCancer = prediction[0] > 0.5; // Example threshold for classification

    const response = {
      id: uuidv4(),
      result: isCancer ? "Cancer" : "Non-cancer",
      suggestion: isCancer
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.",
      createdAt: new Date().toISOString(),
    };

    res.status(200).json({
      status: "success",
      message: "Model is predicted successfully",
      data: response,
    });
  } catch (error) {
    console.error(error);

    res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

app.use((err, req, res, next) => {
  res.status(400).json({
    status: "fail",
    message: err.message || "Terjadi kesalahan dalam melakukan prediksi",
  });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
