import { useEffect, useState, type ChangeEvent } from 'react'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function onPick(e: ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files?.[0]
    e.target.value = '' // allow picking the same file again
    if (!picked) return
    setNotice(null)
    setFile(picked)
  }

  function translate() {
    // Placeholder: the in-browser model / translation gets wired up here later.
    setNotice('Translation is not connected yet.')
  }

  return (
    <main className="app">
      <h1>Hieroglyph Translator</h1>
      <p className="subtitle">Photograph an inscription and get it in English.</p>

      <div className="row">
        <label className="btn">
          Take Photo
          <input type="file" accept="image/*" capture="environment" onChange={onPick} />
        </label>
        <label className="btn btn-alt">
          From Gallery
          <input type="file" accept="image/*" onChange={onPick} />
        </label>
      </div>

      {previewUrl ? (
        <>
          <img className="preview" src={previewUrl} alt="Selected inscription" />
          <button className="btn btn-wide" onClick={translate}>Translate</button>
          {notice && <p className="notice">{notice}</p>}
        </>
      ) : (
        <div className="placeholder">
          <span className="placeholder-glyph">𓂀</span>
          <span>No image yet</span>
        </div>
      )}
    </main>
  )
}
