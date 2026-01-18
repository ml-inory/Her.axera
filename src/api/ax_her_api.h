/**
 * @file ax_template_api.h
 * @brief AX template API header - C-compatible interface for template ASR system
 * @note This header provides a C interface to the template speech recognition system
 */

#ifndef _AX_TEMPLATE_API_H_
#define _AX_TEMPLATE_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#define AX_TEMPLATE_API __attribute__((visibility("default")))

/**
 * @brief Opaque handle type for template ASR context
 * 
 * This handle encapsulates all internal state of the template ASR system.
 * The actual implementation is hidden from C callers to maintain ABI stability.
 */
typedef void* AX_TEMPLATE_HANDLE;

/**
 * @brief Initialize the template ASR system with specific configuration
 * 
 * Creates and initializes a new template ASR context with the specified
 * model type, model path, and language. This function loads the appropriate
 * models, configures the recognizer, and prepares it for speech recognition.
 * 
 * @param model_type Type of template model to use (e.g., "tiny", "base", "small", "medium", "large")
 *                   or custom model identifier
 * @param model_path Directory path where model files are stored
 *                   Model files are expected to be in the format:
 *                   - {model_path}/{model_type}/{model_type}-encoder.axmodel
 *                   - {model_path}/{model_type}/{model_type}-decoder.axmodel
 *                   - {model_path}/{model_type}/{model_type}-tokens.txt
 *                   - {model_path}/{model_type}/{model_type}_config.json
 * @param language Language code for recognition (e.g., "en", "zh", "ja", "ko")
 *                 Use "auto" for automatic language detection if supported
 * 
 * @return AX_TEMPLATE_HANDLE Opaque handle to the initialized template context,
 *         or NULL if initialization fails
 * 
 * @note The caller is responsible for calling AX_TEMPLATE_Uninit() to free
 *       resources when the handle is no longer needed.
 * @note If language is not supported by the model, the function may fall back
 *       to a default language or return NULL.
 * @example
 *   // Initialize English recognition with base model
 *   AX_TEMPLATE_HANDLE handle = AX_TEMPLATE_Init("base", "../models-ax650", "en");
 *   
 */
AX_TEMPLATE_API AX_TEMPLATE_HANDLE AX_TEMPLATE_Init(const char* model_type, const char* model_path, const char* language);

/**
 * @brief Deinitialize and release template ASR resources
 * 
 * Cleans up all resources associated with the template context, including
 * unloading models, freeing memory, and releasing hardware resources.
 * 
 * @param handle template context handle obtained from AX_TEMPLATE_Init()
 * 
 * @warning After calling this function, the handle becomes invalid and
 *          should not be used in any subsequent API calls.
 */
AX_TEMPLATE_API void AX_TEMPLATE_Uninit(AX_TEMPLATE_HANDLE handle);

/**
 * @brief Perform speech recognition and return dynamically allocated string
 * 
 * @param handle template context handle
 * @param wav_file Path to the input 16k pcmf32 WAV audio file
 * @param result Pointer to receive the allocated result string
 * 
 * @return int Status code (0 = success, <0 = error)
 * 
 * @note The returned string is allocated with malloc() and must be freed
 *       by the caller using free() when no longer needed.
 */
AX_TEMPLATE_API int AX_TEMPLATE_RunFile(AX_TEMPLATE_HANDLE handle, 
                   const char* wav_file, 
                   char** result);

/**
 * @brief Perform speech recognition and return dynamically allocated string
 * 
 * @param handle template context handle
 * @param pcm_data 16k Mono PCM f32 data, range from -1.0 to 1.0
 * @param num_samples Sample num of PCM data
 * @param result Pointer to receive the allocated result string
 * 
 * @return int Status code (0 = success, <0 = error)
 * 
 * @note The returned string is allocated with malloc() and must be freed
 *       by the caller using free() when no longer needed.
 */
AX_TEMPLATE_API int AX_TEMPLATE_RunPCM(AX_TEMPLATE_HANDLE handle, 
                   float* pcm_data, 
                   int num_samples,
                   char** result);                   

#ifdef __cplusplus
}
#endif

#endif // _AX_TEMPLATE_API_H_