export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();

  try {
    const { imageBase64, mediaType, category } = req.body;
    if (!imageBase64 || !category) {
      return res.status(400).json({ error: 'Missing imageBase64 or category' });
    }

    const apiKey = process.env.HF_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: 'HF_API_KEY not configured in Vercel environment variables' });
    }

    // Convert base64 to binary buffer for HF API
    const imageBuffer = Buffer.from(imageBase64, 'base64');

    // Model selection based on category
    const modelMap = {
      xray:    'nickmuchi/vit-base-patch16-224-medmnist-chest',
      skin:    'nickmuchi/efficientnet-b4-skin-lesions',
      eye:     'nickmuchi/vit-base-patch16-224-medmnist-retina',
      mri:     'nickmuchi/vit-base-patch16-224-medmnist-tissue',
      micro:   'nickmuchi/vit-base-patch16-224-medmnist-blood',
      general: 'nickmuchi/vit-base-patch16-224-medmnist-chest'
    };

    const model = modelMap[category] || modelMap.general;

    // Call Hugging Face Inference API
    const hfResponse = await fetch(
      `https://router.huggingface.co/hf-inference/models/${model}`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': mediaType || 'image/jpeg'
        },
        body: imageBuffer
      }
    );

    if (!hfResponse.ok) {
      const errText = await hfResponse.text();
      // Model might be loading — return friendly error
      if (hfResponse.status === 503) {
        return res.status(503).json({ error: 'Model is loading, please retry in 20 seconds' });
      }
      return res.status(500).json({ error: `HF API error: ${errText}` });
    }

    const predictions = await hfResponse.json();

    // predictions is array like: [{label: "...", score: 0.95}, ...]
    if (!Array.isArray(predictions) || predictions.length === 0) {
      return res.status(500).json({ error: 'No predictions returned from model' });
    }

    // Build structured medical report from DL predictions
    const result = buildMedicalReport(predictions, category, model);
    return res.status(200).json(result);

  } catch (err) {
    console.error('Analysis error:', err);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
}

function buildMedicalReport(predictions, category, model) {
  // Sort predictions by confidence
  const sorted = [...predictions].sort((a, b) => b.score - a.score);
  const top = sorted[0];
  const topLabel = top.label;
  const topScore = Math.round(top.score * 100);

  // Determine severity based on label and confidence
  const dangerousLabels = [
    'malignant', 'cancer', 'tumor', 'carcinoma', 'melanoma',
    'pneumonia', 'effusion', 'edema', 'infiltration', 'mass',
    'nodule', 'fibrosis', 'emphysema', 'atelectasis', 'consolidation',
    'hemorrhage', 'fracture', 'abnormal', 'positive', 'disease'
  ];

  const warningLabels = [
    'benign', 'mild', 'early', 'suspicious', 'irregular',
    'opacity', 'cardiomegaly', 'pleural', 'retinopathy'
  ];

  const labelLower = topLabel.toLowerCase();
  let severity = 'normal';
  let severityScore = 10;

  if (dangerousLabels.some(d => labelLower.includes(d))) {
    severity = topScore > 80 ? 'high' : 'medium';
    severityScore = topScore > 80 ? Math.round(60 + topScore * 0.4) : Math.round(35 + topScore * 0.4);
  } else if (warningLabels.some(w => labelLower.includes(w))) {
    severity = 'low';
    severityScore = Math.round(20 + topScore * 0.3);
  } else {
    severity = 'normal';
    severityScore = Math.round(topScore * 0.15);
  }

  severityScore = Math.min(severityScore, 100);

  // Build findings from all predictions
  const findings = sorted.slice(0, 4).map(p => {
    const plabel = p.label.toLowerCase();
    const pct = Math.round(p.score * 100);
    let level = 'normal';
    if (dangerousLabels.some(d => plabel.includes(d)) && pct > 40) level = 'danger';
    else if (warningLabels.some(w => plabel.includes(w)) && pct > 30) level = 'warning';
    return {
      level,
      text: `${formatLabel(p.label)}: ${pct}% confidence`
    };
  });

  // Category-specific context
  const categoryInfo = {
    xray:    { name: 'Chest X-Ray', arch: 'ViT-Base (Vision Transformer)', dataset: 'MedMNIST ChestMNIST' },
    skin:    { name: 'Skin Lesion', arch: 'EfficientNet-B4', dataset: 'ISIC Skin Lesion Dataset' },
    eye:     { name: 'Retinal Scan', arch: 'ViT-Base (Vision Transformer)', dataset: 'MedMNIST RetinaMNIST' },
    mri:     { name: 'Tissue/MRI', arch: 'ViT-Base (Vision Transformer)', dataset: 'MedMNIST TissueMNIST' },
    micro:   { name: 'Blood Microscopy', arch: 'ViT-Base (Vision Transformer)', dataset: 'MedMNIST BloodMNIST' },
    general: { name: 'Medical Image', arch: 'ViT-Base (Vision Transformer)', dataset: 'MedMNIST ChestMNIST' }
  };

  const info = categoryInfo[category] || categoryInfo.general;

  const urgency = severityScore > 70 ? 'urgent' : severityScore > 40 ? 'soon' : severityScore > 20 ? 'routine' : 'routine';

  const possibleConditions = sorted
    .filter(p => p.score > 0.1)
    .map(p => formatLabel(p.label))
    .slice(0, 3);

  return {
    summary: `Deep learning analysis of ${info.name} using ${info.arch} trained on ${info.dataset}. Primary classification: ${formatLabel(topLabel)} with ${topScore}% confidence. Analysis based on ${sorted.length} output classes.`,
    primaryFinding: `${formatLabel(topLabel)} — ${topScore}% confidence`,
    severity,
    severityScore,
    confidence: topScore,
    findings,
    possibleConditions,
    recommendation: getRecommendation(severity, topLabel, category),
    urgency,
    modelUsed: model,
    architecture: info.arch,
    dataset: info.dataset,
    allPredictions: sorted.map(p => ({
      label: formatLabel(p.label),
      score: Math.round(p.score * 100)
    }))
  };
}

function formatLabel(label) {
  return label
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

function getRecommendation(severity, label, category) {
  if (severity === 'high') {
    return 'Urgent medical consultation recommended. Please visit a specialist immediately for further evaluation and confirmation of findings.';
  } else if (severity === 'medium') {
    return 'Medical follow-up advised within the next few days. Schedule an appointment with a relevant specialist for detailed examination.';
  } else if (severity === 'low') {
    return 'Monitor the condition and schedule a routine checkup. Discuss findings with your primary care physician at your next visit.';
  }
  return 'No immediate action required. Continue routine health monitoring and regular checkups as advised by your healthcare provider.';
}
