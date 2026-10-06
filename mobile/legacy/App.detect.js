import { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';

// ── Change this to your PC's local IP ──────────────────────────────────────
const API_URL = 'http://192.168.0.253:8000/detect';
// ───────────────────────────────────────────────────────────────────────────

export default function App() {
  const [loading, setLoading]          = useState(false);
  const [annotatedImage, setAnnotated] = useState(null);
  const [result, setResult]            = useState(null);

  async function pickImage(useCamera) {
    const perm = useCamera
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (perm.status !== 'granted') {
      Alert.alert('Permission needed', 'Please allow access to continue.');
      return;
    }

    const picked = useCamera
      ? await ImagePicker.launchCameraAsync({ quality: 0.85 })
      : await ImagePicker.launchImageLibraryAsync({ quality: 0.85 });

    if (picked.canceled) return;
    sendToAPI(picked.assets[0]);
  }

  async function sendToAPI(asset) {
    setLoading(true);
    setResult(null);
    setAnnotated(null);

    try {
      const form = new FormData();
      form.append('file', {
        uri:  asset.uri,
        name: 'photo.jpg',
        type: 'image/jpeg',
      });

      const res  = await fetch(API_URL, { method: 'POST', body: form });
      const data = await res.json();

      setAnnotated(`data:image/jpeg;base64,${data.annotated_image}`);
      setResult(data);
    } catch (e) {
      Alert.alert(
        'Connection error',
        'Make sure your PC server is running and both devices are on the same WiFi.\n\n' + e.message,
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={s.safe}>
      <ScrollView contentContainerStyle={s.scroll}>

        <Text style={s.title}>Hieroglyph Translator</Text>

        <View style={s.row}>
          <TouchableOpacity style={s.btn} onPress={() => pickImage(true)}>
            <Text style={s.btnText}>Take Photo</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[s.btn, s.btnAlt]} onPress={() => pickImage(false)}>
            <Text style={s.btnText}>Pick from Gallery</Text>
          </TouchableOpacity>
        </View>

        {loading && (
          <View style={s.center}>
            <ActivityIndicator size="large" color="#c8a96e" />
            <Text style={s.hint}>Analysing hieroglyphs…</Text>
          </View>
        )}

        {annotatedImage && (
          <Image source={{ uri: annotatedImage }} style={s.image} resizeMode="contain" />
        )}

        {result && (
          <View style={s.results}>
            <Text style={s.label}>
              {result.count} hieroglyph{result.count !== 1 ? 's' : ''} detected
            </Text>

            {result.transliteration ? (
              <>
                <Text style={s.sectionHead}>Transliteration</Text>
                <Text style={s.translit}>{result.transliteration}</Text>
              </>
            ) : null}

            <Text style={s.sectionHead}>Reading order</Text>
            {result.detections.map((d, i) => (
              <View key={i} style={s.card}>
                <Text style={s.cardNum}>{i + 1}</Text>
                <View style={s.cardBody}>
                  <Text style={s.cardName}>{d.class_name.replace(/_/g, ' ')}</Text>
                  <Text style={s.cardDetail}>
                    {d.phonetic !== '—' ? `Phonetic: ${d.phonetic}   ` : ''}
                    {d.gardiner !== '—' ? `Gardiner: ${d.gardiner}` : ''}
                  </Text>
                  <Text style={s.cardMeaning}>{d.meaning}</Text>
                </View>
                <Text style={s.cardConf}>{Math.round(d.confidence * 100)}%</Text>
              </View>
            ))}
          </View>
        )}

      </ScrollView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  safe:        { flex: 1, backgroundColor: '#1a1207' },
  scroll:      { padding: 20, paddingBottom: 40 },
  title:       { fontSize: 24, fontWeight: '700', color: '#c8a96e',
                 textAlign: 'center', marginBottom: 24, letterSpacing: 1 },
  row:         { flexDirection: 'row', gap: 12, marginBottom: 24 },
  btn:         { flex: 1, backgroundColor: '#c8a96e', borderRadius: 10,
                 paddingVertical: 14, alignItems: 'center' },
  btnAlt:      { backgroundColor: '#5a3e1b' },
  btnText:     { color: '#1a1207', fontWeight: '700', fontSize: 15 },
  center:      { alignItems: 'center', marginVertical: 20 },
  hint:        { color: '#c8a96e', marginTop: 10 },
  image:       { width: '100%', height: 300, borderRadius: 10,
                 marginBottom: 20, backgroundColor: '#2a1f0e' },
  results:     { gap: 6 },
  label:       { color: '#c8a96e', fontWeight: '700', fontSize: 16,
                 marginBottom: 8 },
  sectionHead: { color: '#9a7a4a', fontSize: 13, fontWeight: '600',
                 textTransform: 'uppercase', letterSpacing: 1,
                 marginTop: 12, marginBottom: 4 },
  translit:    { color: '#f0ddb0', fontSize: 20, fontWeight: '300',
                 letterSpacing: 2, marginBottom: 8 },
  card:        { flexDirection: 'row', alignItems: 'center',
                 backgroundColor: '#2a1f0e', borderRadius: 8,
                 padding: 12, gap: 10, marginBottom: 6 },
  cardNum:     { color: '#9a7a4a', fontSize: 13, fontWeight: '700', width: 22 },
  cardBody:    { flex: 1, gap: 2 },
  cardName:    { color: '#f0ddb0', fontWeight: '600', fontSize: 15 },
  cardDetail:  { color: '#9a7a4a', fontSize: 12 },
  cardMeaning: { color: '#c8a96e', fontSize: 13 },
  cardConf:    { color: '#5a7a4a', fontSize: 12, fontWeight: '600' },
});
