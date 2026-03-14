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

    const categoryContext = {
      general: "You are an expert radiologist and medical image analyst.",
      xray:    "You are an expert radiologist specializing in X-ray interpretation.",
      skin:    "You are an expert dermatologist and dermoscopy specialist.",
      eye:     "You are an expert ophthalmologist specializing in retinal imaging.",
      mri:     "You are an expert neuroradiologist specializing in MRI and CT imaging.",
      micro:   "You are an expert pathologist and microbiologist."
    };

    const context = categoryContext[category] || categoryContext.general;

    const prompt = `${context}

Analyze this medical image and respond ONLY with a valid JSON object. No markdown, no backticks, no explanation — just raw JSON exactly like this:

{"summary":"2-3 sentence overview","primaryFinding":"main finding","severity":"normal","severityScore":10,"confidence":85,"findings":[{"level":"normal","text":"finding 1"},{"level":"warning","text":"finding 2"}],"possibleConditions":["condition1","condition2"],"recommendation":"actionable recommendation","urgency":"routine"}

severity must be: normal, low, medium, or high
urgency must be: routine, soon, urgent, or emergency
level must be: normal, warning, or danger`;

    // Use :fastest to auto-select available provider
    const hfResponse = await fetch(
      'https://router.huggingface.co/v1/chat/completions',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'Qwen/Qwen2.5-VL-7B-Instruct:fastest',
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'image_url',
                  image_url: {
                    url: `data:${mediaType || 'image/jpeg'};base64,${imageBase64}`
                  }
                },
                {
                  type: 'text',
                  text: prompt
                }
              ]
            }
          ],
          max_tokens: 1000,
          temperature: 0.3
        })
      }
    );

    if (!hfResponse.ok) {
      const errText = await hfResponse.text();
      if (hfResponse.status === 503) {
        return res.status(503).json({ error: 'Model is loading, please retry in 20 seconds' });
      }
      return res.status(500).json({ error: `HF API error: ${errText}` });
    }

    const data = await hfResponse.json();
    let text = data.choices?.[0]?.message?.content || '';
    text = text.replace(/```json|```/g, '').trim();

    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return res.status(500).json({ error: 'Model returned invalid response. Try again.' });
    }

    const result = JSON.parse(jsonMatch[0]);

    result.modelUsed = 'Qwen/Qwen2.5-VL-7B-Instruct';
    result.architecture = 'Qwen2.5-VL (7B Vision Language Model)';
    result.dataset = 'Multimodal Medical + General Dataset';
    result.allPredictions = [
      { label: result.primaryFinding, score: result.confidence || 85 },
      ...(result.possibleConditions || []).map((c, i) => ({
        label: c,
        score: Math.max(10, (result.confidence || 85) - (i + 1) * 15)
      }))
    ];

    return res.status(200).json(result);

  } catch (err) {
    console.error('Analysis error:', err);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
}
