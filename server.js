const express = require("express");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { Storage } = require("@google-cloud/storage");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());

// Inisialisasi Google Cloud Storage client
const storage = new Storage();
const bucketName = "ml-model-bucket-4";  // Nama bucket Anda
const modelPath = "models/model.json";  // Path model.json
const modelShardsPrefix = "models/group1-shard";  // Prefix untuk shard

const upload = multer({
  limits: { fileSize: 1000000 },
  fileFilter(req, file, cb) {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("File must be an image"));
    }
    cb(null, true);
  },
});

// Fungsi untuk mengunduh file dari Google Cloud Storage
async function downloadFileFromGCS(gcsFilePath, localFilePath) {
  const options = {
    destination: localFilePath,
  };

  await storage
    .bucket(bucketName)
    .file(gcsFilePath)
    .download(options);

  console.log(`Downloaded ${gcsFilePath} to ${localFilePath}`);
}

// Memuat Model dari GCS
let model;

async function loadModel() {
  try {
    // Unduh file model.json
    const tempModelPath = path.join(__dirname, "model.json");
    await downloadFileFromGCS(modelPath, tempModelPath);

    // Unduh semua shard model
    const shardFiles = [
      "group1-shard1of4.bin",
      "group1-shard2of4.bin",
      "group1-shard3of4.bin",
      "group1-shard4of4.bin",
    ];

    for (const shard of shardFiles) {
      const tempShardPath = path.join(__dirname, shard);
      await downloadFileFromGCS(`${modelShardsPrefix}${shard}`, tempShardPath);
    }

    // Memuat model dari file
    model = await tf.loadGraphModel(`file://${tempModelPath}`);
    console.log("Model loaded successfully");

    // Menghapus file sementara setelah dimuat
    fs.unlinkSync(tempModelPath);
    shardFiles.forEach((shard) => fs.unlinkSync(path.join(__dirname, shard)));

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
    const tensor = tf.node.decodeImage(buffer, 3);
    const resized = tf.image.resizeBilinear(tensor, [224, 224]);
    const normalized = resized.div(255).expandDims(0); 

    const prediction = await model.predict(normalized).dataSync(); 
    const isCancer = prediction[0] > 0.5;

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

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on http://IP:${PORT}`);
});
