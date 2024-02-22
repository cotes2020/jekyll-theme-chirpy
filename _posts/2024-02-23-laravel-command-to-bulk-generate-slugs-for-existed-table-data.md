---
published: true
date: 2023-10-07
title: Laravel command to bulk generate slugs for existed table data
---
Create a command in Laravel:

    php artisan make:command CommandName
    

Example with `vocabularies` table:

    <?php
    
    namespace App\Console\Commands;
    
    use Illuminate\Console\Command;
    use App\Models\Vocabulary;
    use Illuminate\Support\Str;
    use Illuminate\Support\Facades\DB;
    
    class AddVocabularySlug extends Command
    {
        /**
         * The name and signature of the console command.
         *
         * @var string
         */
        protected $signature = 'voc:update-slug';
    
        /**
         * The console command description.
         *
         * @var string
         */
        protected $description = '';
    
        /**
         * Execute the console command.
         */
        public function handle()
        {
            $vocabularies = Vocabulary::whereNull('slug')->get()->map(function ($vocabulary) {
                echo "Mapping vocabulary id: {$vocabulary->id}\n";
                return [
                    'id' => $vocabulary->id,
                    'slug' => Str::substr(Str::slug($vocabulary->mean), 0, 255)
                ];
            })->toArray();
            echo "Total vocabularies: " . count($vocabularies) . "\n";
            $this->updateBatch('vocabularies', $vocabularies);
        }
    
        function updateBatch($tableName = "", $multipleData = array())
        {
    
            if ($tableName && !empty($multipleData)) {
    
                // column or fields to update
                $updateColumn = array_keys($multipleData[0]);
                $referenceColumn = $updateColumn[0]; //e.g id
                unset($updateColumn[0]);
                $whereIn = "";
    
                $q = "UPDATE " . $tableName . " SET ";
                foreach ($updateColumn as $uColumn) {
                    $q .=  $uColumn . " = CASE ";
    
                    foreach ($multipleData as $data) {
                        $q .= "WHEN " . $referenceColumn . " = " . $data[$referenceColumn] . " THEN '" . $data[$uColumn] . "' ";
                    }
                    $q .= "ELSE " . $uColumn . " END, ";
                }
                foreach ($multipleData as $data) {
                    echo "Processing vocabulary id: {$data[$referenceColumn]}\n";
    
                    $whereIn .= "'" . $data[$referenceColumn] . "', ";
                }
                $q = rtrim($q, ", ") . " WHERE " . $referenceColumn . " IN (" .  rtrim($whereIn, ', ') . ")";
    
                echo "Updating\n";
    
                // Update  
                return DB::update(DB::raw($q));
            } else {
                return false;
            }
        }
    }