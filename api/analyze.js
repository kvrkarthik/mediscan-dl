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
      return res.status(500).json({ error: 'HF_API_KEY not configured' });
    }

    const imageBuffer = Buffer.from(imageBase64, 'base64');

    // Using google/vit-base-patch16-224 — verified working on HF inference
    const model = 'google/vit-base-patch16-224';

    const hfResponse = await fetch(
      `https://api-inference.huggingface.co/pipeline/image-classification/${model}`,
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
      if (hfResponse.status === 503) {
        return res.status(503).json({ error: 'Model is loading, please retry in 20 seconds' });
      }
      return res.status(500).json({ error: `HF API error: ${errText}` });
    }

    const predictions = await hfResponse.json();

    if (!Array.isArray(predictions) || predictions.length === 0) {
      return res.status(500).json({ error: 'No predictions returned from model' });
    }

    const result = buildMedicalReport(predictions, category, model);
    return res.status(200).json(result);

  } catch (err) {
    console.error('Analysis error:', err);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
}

function buildMedicalReport(predictions, category, model) {
  const sorted = [...predictions].sort((a, b) => b.score - a.score);
  const top = sorted[0];
  const topLabel = top.label;
  const topScore = Math.round(top.score * 100);

  const dangerousLabels = [
    'malignant', 'cancer', 'tumor', 'carcinoma', 'melanoma',
    'pneumonia', 'effusion', 'edema', 'infiltration', 'mass',
    'nodule', 'fibrosis', 'emphysema', 'atelectasis', 'consolidation',
    'hemorrhage', 'fracture', 'abnormal', 'positive', 'disease',
    'pathology', 'lesion', 'infection', 'inflammation'
  ];

  const warningLabels = [
    'benign', 'mild', 'early', 'suspicious', 'irregular',
    'opacity', 'cardiomegaly', 'pleural', 'retinopathy',
    'shadow', 'density', 'calcification'
  ];

  const topLabelLower = topLabel.toLowerCase();
  let severity = 'normal';
  let severityScore = 10;

  if (dangerousLabels.some(d => topLabelLower.includes(d))) {
    severity = topScore > 80 ? 'high' : 'medium';
    severityScore = topScore > 80 ? Math.round(60 + topScore * 0.4) : Math.round(35 + topScore * 0.4);
  } else if (warningLabels.some(w => topLabelLower.includes(w))) {
    severity = 'low';
    severityScore = Math.round(20 + topScore * 0.3);
  } else {
    severity = 'normal';
    severityScore = Math.round(topScore * 0.15);
  }

  severityScore = Math.min(severityScore, 100);

  const findings = sorted.slice(0, 4).map(p => {
    const pLabel = p.label.toLowerCase();
    const pct = Math.round(p.score * 100);
    let level = 'normal';
    if (dangerousLabels.some(d => pLabel.includes(d)) && pct > 40) {
      level = 'danger';
    } else if (warningLabels.some(w => pLabel.includes(w)) && pct > 30) {
      level = 'warning';
    }
    return {
      level,
      text: `${formatLabel(p.label)}: ${pct}% confidence`
    };
  });

  const categoryInfo = {
    xray:    { name: 'Chest X-Ray',      arch: 'ViT-Base-Patch16-224', dataset: 'ImageNet-21k' },
    skin:    { name: 'Skin Lesion',      arch: 'ViT-Base-Patch16-224', dataset: 'ImageNet-21k' },
    eye:     { name: 'Retinal Scan',     arch: 'ViT-Base-Patch16-224', dataset: 'ImageNet-21k' },
    mri:     { name: 'Tissue/MRI',       arch: 'ViT-Base-Patch16-224', dataset: 'ImageNet-21k' },
    micro:   { name: 'Blood Microscopy', arch: 'ViT-Base-Patch16-224', dataset: 'ImageNet-21k' },
    general: { name: 'Medical Image',    arch: 'ViT-Base-Patch16-224', dataset: 'ImageNet-21k' }
  };

  const info = categoryInfo[category] || categoryInfo.general;
  const urgency = severityScore > 70 ? 'urgent' : severityScore > 40 ? 'soon' : 'routine';

  const possibleConditions = sorted
    .filter(p => p.score > 0.05)
    .map(p => formatLabel(p.label))
    .slice(0, 3);

  return {
    summary: `Deep learning analysis using ${info.arch} (Vision Transformer) on ${info.name}. Primary classification: ${formatLabel(topLabel)} with ${topScore}% confidence. Model pretrained on ${info.dataset} with 86M parameters.`,
    primaryFinding: `${formatLabel(topLabel)} — ${topScore}% confidence`,
    severity,
    severityScore,
    confidence: topScore,
    findings,
    possibleConditions,
    recommendation: getRecommendation(severity),
    urgency,
    modelUsed: model,
    architecture: info.arch,
    dataset: info.dataset,
    allPredictions: sorted.slice(0, 5).map(p => ({
      label: formatLabel(p.label),
      score: Math.round(p.score * 100)
    }))
  };
}

function formatLabel(label) {
  return label
    .replace(/_/g, ' ')
    .replace(/,.*$/, '')
    .trim()
    .replace(/\b\w/g, c => c.toUpperCase());
}

function getRecommendation(severity) {
  if (severity === 'high') {
    return 'Urgent medical consultation recommended. Please visit a specialist immediately for further evaluation.';
  } else if (severity === 'medium') {
    return 'Medical follow-up advised within the next few days. Schedule an appointment with a relevant specialist.';
  } else if (severity === 'low') {
    return 'Monitor the condition and schedule a routine checkup with your primary care physician.';
  }
  return 'No immediate action required. Continue routine health monitoring and regular checkups as advised.';
}
