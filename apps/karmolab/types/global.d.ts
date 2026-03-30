import type { KarmoLabImageBatchAPI, KarmoLabImageConvertAPI, RandomGenTopic } from './karmolab';

export {};

declare global {
  interface Window {
    KarmoLabImageConvert?: KarmoLabImageConvertAPI;
    KarmoLabImageBatch?: KarmoLabImageBatchAPI;
    RANDOMGEN_TOPICS?: RandomGenTopic[];
  }
}
