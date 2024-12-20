const express = require("express");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { Storage } = require("@google-cloud/storage");
const admin = require("firebase-admin");

const app = express();
const PORT = 3000;

app.use(bodyParser.json());

// Inisialisasi Google Cloud Storage client
const storage = new Storage();
const bucketName = "ml-model-bucket-4";
const modelPath = "models/model.json";

const serviceAccount = require("./serviceAccountKey.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const db = admin.firestore();

const upload = multer({
  dest: "uploads/",
  limits: { fileSize: 1000000 }, // Batas ukuran file 1MB
  fileFilter(req, file, cb) {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("Terjadi kesalahan dalam melakukan prediksi"));
    }
    cb(null, true);
  },
});

async function downloadFileFromGCS(gcsFilePath, localFilePath) {
  const options = {
    destination: localFilePath,
  };

  await storage.bucket(bucketName).file(gcsFilePath).download(options);

  console.log(`Downloaded ${gcsFilePath} to ${localFilePath}`);
}

// Memuat Model dari GCS
let model;

async function loadModel() {
  try {
    const tempModelPath = path.join(__dirname, "models/model.json");
    await downloadFileFromGCS(modelPath, tempModelPath);

    const shardFiles = [
      "group1-shard1of4.bin",
      "group1-shard2of4.bin",
      "group1-shard3of4.bin",
      "group1-shard4of4.bin",
    ];

    for (const shard of shardFiles) {
      const tempShardPath = path.join(__dirname, `models/${shard}`);
      await downloadFileFromGCS(`models/${shard}`, tempShardPath);
    }

    model = await tf.loadGraphModel(`file://${tempModelPath}`);
    console.log("Model loaded successfully");
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

// Panggil fungsi loadModel saat aplikasi dimulai
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
    const uint8Array = new Uint8Array(buffer);

    const tensor = tf.node.decodeImage(uint8Array, 3);
    const resized = tf.image.resizeBilinear(tensor, [224, 224]);
    const normalized = resized.div(255).expandDims(0);

    const prediction = await model.predict(normalized);
    const predictionData = await prediction.data();
    const isCancer = predictionData[0] > 0.58;

    if (predictionData[0] < 0.5) {
      return res.status(400).json({
        status: "fail",
        message: "Terjadi kesalahan dalam melakukan prediksi",
      });
    }

    const response = {
      id: uuidv4(),
      result: isCancer ? "Cancer" : "Non-cancer",
      suggestion: isCancer
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.",
      createdAt: new Date().toISOString(),
    };

    await db.collection("predictions").doc(response.id).set(response);

    res.status(201).json({
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

// Middleware untuk menangani error file yang melebihi batas ukuran
app.use((err, req, res, next) => {
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({
      status: "fail",
      message: "Payload content length greater than maximum allowed: 1000000",
    });
  }

  res.status(400).json({
    status: "fail",
    message: "Terjadi kesalahan dalam melakukan prediksi",
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://IP:${PORT}`);
});
