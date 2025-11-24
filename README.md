# Diyabet Tahmini İçin Melez Derinöğrenme Modeli Autoencoder ve MLP Uygulaması
Bu çalışma, bireylerin sağlık verilerine dayalı olarak diyabet durumlarını tahmin etmek amacıyla Hacettepe Üniversitesi İstatistik Bölümü'nde IST493 Seminer Dersi kapsamında geliştirilmiştir. Projede, veri setindeki dengesizliği gidermek için SMOTE yöntemi uygulanmış, ardından boyut indirgeme için bir Autoencoder ve sınıflandırma için bir Çok Katmanlı Algılayıcı (MLP)  içeren hibrit bir derin öğrenme mimarisi kullanılmıştır.

# 📝 1. Veri Ön İşleme (Data Preprocessing)

Veri seti, yaş, BMI, fiziksel ve zihinsel sağlık durumu, sigara/alkol kullanımı, tansiyon ve kolesterol gibi 21 bağımsız değişken ve Diabetes_binary adlı hedef değişkeni (0: Sağlıklı, 1: Diyabet Hastası) içermektedir.

## 1.1 Kütüphane Yükleme ve Veri İçe Aktarma
Analiz için gerekli olan temel Python kütüphaneleri (NumPy, Pandas, TensorFlow, scikit-learn ve imblearn) yüklenmiştir.

<img width="557" height="233" alt="image" src="https://github.com/user-attachments/assets/343fe8af-27db-4c7d-b103-a8227bcf9804" />

<img width="566" height="190" alt="image" src="https://github.com/user-attachments/assets/0aa090dd-efa7-4ad3-a404-bba4497f4f63" />

<img width="314" height="100" alt="image" src="https://github.com/user-attachments/assets/5e806b63-c6df-467c-8081-b888f27e67d9" />

<img width="640" height="316" alt="image" src="https://github.com/user-attachments/assets/50259695-7c8c-46a0-aa14-d12f2ad69ccf" />

Değişkenler arası ilişki olup olmadığını kontrol etmek için ısı haritası çizdirilmiş ve herhangi iki değişken arasında ciddi bir ilişkiye rastlanmamıştır. 

<img width="881" height="169" alt="image" src="https://github.com/user-attachments/assets/c524a8eb-3005-4a51-8438-576f738e0433" />

<img width="945" height="724" alt="image" src="https://github.com/user-attachments/assets/6b06f325-03bb-434d-9a0b-86d6a445a2a5" />

## 1.2 Hedef Değişken Analizi ve Dengelenmesi

Hedef değişkenin (Diabetes_binary) dağılımında ciddi bir dengesizlik tespit edilmiştir (0: 218334, 1: 35346). Sınıflandırma modelinin diyabetli bireyleri doğru tahmin etme yeteneğini artırmak için SMOTE yöntemi kullanılarak veri seti dengelenmiştir.

<img width="620" height="188" alt="image" src="https://github.com/user-attachments/assets/ac1b4f6a-48d8-4967-b35b-0008675b784f" />

<img width="945" height="148" alt="image" src="https://github.com/user-attachments/assets/e38338f9-905b-4ee0-acc0-a37d73627d60" />

# 🧠 2. Model Eğitimi (Model Training)

Diyabet tahminini gerçekleştirmek için Autoencoder ve MLP'den oluşan hibrit bir derin öğrenme mimarisi uygulanmıştır.

## 2.1 Autoencoder ile Özellik Çıkarımı

Autoencoder, giriş verisinin boyutunu düşürerek encoding_dim = 10 daha kompakt ve anlamlı bir özet temsilini çıkarmak için kullanılmıştır.

<img width="863" height="563" alt="image" src="https://github.com/user-attachments/assets/7e1aa200-cbdb-41be-8527-b8233e71f4d1" />

## 2.2 MLP Modeli ile Sınıflandırma 

Autoencoder'dan elde edilen 10 boyutlu özet temsil, MLP modeline girdi olarak verilmiştir. Modelde iki gizli katman 64 ve 32 nöronlu kullanılmış ve çıkış katmanında ikili sınıflandırma için softmax aktivasyon fonksiyonu kullanılmıştır.

<img width="817" height="447" alt="image" src="https://github.com/user-attachments/assets/536e4ca7-2359-4fa4-8709-a605519218bd" />

 # ✅ 3. Model Değerlendirme (Evaluation)
 
Modelin başarısı genel doğruluk (accuracy) ve sınıflar bazında Precision, Recall ve F1-score gibi kritik metriklerle değerlendirilmiştir.

## 3.1 Performans Metrikleri

Modelin test verisi üzerindeki genel doğruluğu %74.06 olarak elde edilmiştir.

<img width="813" height="382" alt="image" src="https://github.com/user-attachments/assets/e55cc178-fb3a-4ec4-9084-2119ad39ab3a" />

### Çıktı Özeti:

<img width="289" height="58" alt="image" src="https://github.com/user-attachments/assets/c45ddd41-6209-4205-871f-c6fdced3a61a" />
<img width="696" height="225" alt="image" src="https://github.com/user-attachments/assets/d8d84a31-4308-40a0-abbd-f75e0e1b98b7" />

### Recall Yorumu:

Diyabetli bireylerin doğru tahmin edilme oranı (Recall değeri) %79 ile oldukça yüksek çıkmıştır. Bu, modelin özellikle pozitif sınıfı (diyabeti) tanıma konusunda etkili olduğunu gösterir.

## 3.2 Karışıklık Matrisi 

Karışıklık matrisi, modelin hangi sınıfları doğru, hangilerini yanlış sınıflandırdığını detaylı olarak göstermiştir.

<img width="790" height="276" alt="image" src="https://github.com/user-attachments/assets/0a76d3cc-0d0a-47e2-8f55-cd231f8c0947" />
<img width="691" height="465" alt="image" src="https://github.com/user-attachments/assets/377b5657-ac26-45fb-a0dd-3371e0243dc3" />

### Matris Yorumu: 

Model, diyabetli bireylerin büyük bir kısmını (%79) ve sağlıklı bireylerin %69'unu doğru sınıflandırmıştır.

## 3.3 ROC Eğrisi ve AUC Skoru
Modelin sınıfları ayırma gücünü daha iyi değerlendirmek için ROC Eğrisi çizilmiş ve altında kalan alan (AUC) hesaplanmıştır.

<img width="860" height="274" alt="image" src="https://github.com/user-attachments/assets/82b9160e-a685-464b-ba49-bc69b840c94e" />
<img width="814" height="365" alt="image" src="https://github.com/user-attachments/assets/46380c95-15c1-4dcb-b11d-4bcb3707cfb3" />
<img width="706" height="521" alt="image" src="https://github.com/user-attachments/assets/0b782209-3049-4ecc-9fd9-8bc6a2a39356" />
<img width="600" height="467" alt="image" src="https://github.com/user-attachments/assets/0cbe9388-9c78-4178-942c-87ebc987e59f" />

### AUC Yorumu:

Elde edilen 0.81'lik AUC değeri, modelin pozitif sınıfı (diyabet hastalarını) rastgele bir sınıflayıcıya göre oldukça iyi bir başarıyla ayırt etme gücüne sahip olduğunu göstermektedir.

# 🚀 Sonuç ve Yorum
Uygulanan SMOTE + Autoencoder + MLP melez mimarisi, diyabet tahmini için güvenilir ve istikrarlı bir performans sergilemiştir. Modelin %74 doğruluk oranı ve özellikle %79'luk yüksek Recall değeri, bu yöntemin diyabet riski taşıyan bireyleri doğru bir şekilde tespit etmede klinik olarak uygulanabilir bir potansiyel sunduğunu göstermektedir.

# 🔗 Kaynakça
Kullanılan ana kaynak ve kütüphane dokümantasyonları:
TensorFlow/Keras Resmi Dokümantasyonu 
imbalanced-learn (SMOTE) Dokümantasyonu 
Hacettepe Üniversitesi Derin Öğrenme Kaynakları

# 🎓 Proje Künyesi
Hazırlayan: Müleyke Önal
Ders Sorumlusu: Prof. Dr. Serpil Aktaş Altunay
Kurum: Hacettepe Üniversitesi, İstatistik Bölümü








