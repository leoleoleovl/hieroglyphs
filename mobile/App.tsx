import { useState } from 'react';
import { Alert, Image, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';

type Source = 'camera' | 'gallery';

export default function App() {
  const [imageUri, setImageUri] = useState<string | null>(null);

  async function pickImage(source: Source) {
    const perm =
      source === 'camera'
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!perm.granted) {
      Alert.alert('Permission needed', 'Please allow access in your phone settings to continue.');
      return;
    }

    const options: ImagePicker.ImagePickerOptions = { mediaTypes: ['images'], quality: 0.85 };
    const picked =
      source === 'camera'
        ? await ImagePicker.launchCameraAsync(options)
        : await ImagePicker.launchImageLibraryAsync(options);

    if (picked.canceled) return;
    setImageUri(picked.assets[0].uri);
  }

  function translate() {
    // Placeholder: the AI-vision translation gets wired up here later.
    Alert.alert('Coming soon', 'Translation is not connected yet.');
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView style={s.safe}>
        <StatusBar style="light" />
        <ScrollView contentContainerStyle={s.scroll}>
          <Text style={s.title}>Hieroglyph Translator</Text>
          <Text style={s.subtitle}>Photograph an inscription and get it in English.</Text>

          <View style={s.row}>
            <TouchableOpacity style={s.btn} onPress={() => pickImage('camera')}>
              <Text style={s.btnText}>Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[s.btn, s.btnAlt]} onPress={() => pickImage('gallery')}>
              <Text style={[s.btnText, s.btnAltText]}>From Gallery</Text>
            </TouchableOpacity>
          </View>

          {imageUri ? (
            <>
              <Image source={{ uri: imageUri }} style={s.image} resizeMode="contain" />
              <TouchableOpacity style={s.translateBtn} onPress={translate}>
                <Text style={s.btnText}>Translate</Text>
              </TouchableOpacity>
            </>
          ) : (
            <View style={s.placeholder}>
              <Text style={s.placeholderGlyph}>𓂀</Text>
              <Text style={s.hint}>No image yet</Text>
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const s = StyleSheet.create({
  safe:             { flex: 1, backgroundColor: '#1a1207' },
  scroll:           { padding: 20, paddingBottom: 40 },
  title:            { fontSize: 26, fontWeight: '700', color: '#c8a96e',
                      textAlign: 'center', marginTop: 12, letterSpacing: 1 },
  subtitle:         { color: '#9a7a4a', textAlign: 'center', marginTop: 6, marginBottom: 24 },
  row:              { flexDirection: 'row', gap: 12, marginBottom: 20 },
  btn:              { flex: 1, backgroundColor: '#c8a96e', borderRadius: 10,
                      paddingVertical: 14, alignItems: 'center' },
  btnAlt:           { backgroundColor: '#5a3e1b' },
  btnText:          { color: '#1a1207', fontWeight: '700', fontSize: 16 },
  btnAltText:       { color: '#f0ddb0' },
  image:            { width: '100%', height: 360, borderRadius: 10,
                      backgroundColor: '#2a1f0e', marginBottom: 16 },
  translateBtn:     { backgroundColor: '#c8a96e', borderRadius: 10,
                      paddingVertical: 16, alignItems: 'center' },
  placeholder:      { height: 360, borderRadius: 10, borderWidth: 2, borderStyle: 'dashed',
                      borderColor: '#5a3e1b', alignItems: 'center', justifyContent: 'center' },
  placeholderGlyph: { fontSize: 64, color: '#5a3e1b' },
  hint:             { color: '#9a7a4a', marginTop: 8 },
});
