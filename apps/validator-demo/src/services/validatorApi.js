export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("profile_id", "default");

  const res = await fetch("http://127.0.0.1:8000/analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Request failed");
  }

  return res.json();
}