<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Support\Dir;
use ArrayObject;
use InvalidArgumentException;
use LogicException;
use Countable;
use IteratorAggregate;

class ClassifiedDirectoryDataset implements Countable,IteratorAggregate,Dataset
{
    use GenericUtils;
    protected $mo;
    protected $path;
    protected $pattern;
    protected $batchSize;
    protected $filter;
    protected $crawler;
    protected $length;
    protected $delimiter;
    protected $enclosure;
    protected $escape;
    protected $filenames;
    protected $maxSteps=0;
    protected $maxDatasetSize=0;

    public function __construct(
        $mo, string $path, array $options=null,
        array &$leftargs=null
        )
    {
        extract($this->extractArgs([
            'pattern'=>null,
            'batch_size'=>32,
            'crawler'=>null,
            'filter'=>null,
            'unclassified'=>false,
        ],$options,$leftargs));
        $this->mo = $mo;
        $this->crawler = $crawler;
        $this->path = $path;
        $this->pattern = $pattern;
        $this->batchSize = $batch_size;
        if($crawler==null) {
            $crawler = new Dir();
        }
        $this->crawler = $crawler;
        $this->filter = $filter;
        $this->unclassified = $unclassified;
    }

    public function setFilter(DatasetFilter $filter) : void
    {
        $this->filter = $filter;
    }

    public function batchSize() : int
    {
        return $this->batchSize;
    }

    public function datasetSize() : int
    {
        return $this->maxDatasetSize;
    }

    public function count()
    {
        return $this->maxSteps;
    }

    protected function getFilenames()
    {
        if($this->filenames===null) {
            $this->filenames = $this->crawler->glob($this->path,$this->pattern);
        }
        return $this->filenames;
    }

    protected function readContents($filename)
    {
        return file_get_contents($filename);
    }

    protected function makeBatchInputs($inputs)
    {
        return $inputs;
    }

    protected function makeBatchTests($tests)
    {
        return $tests;
    }

    public function  getIterator()
    {
        $la = $this->mo->la();
        $filenames = $this->getFilenames();
        $prefixLength = strlen($this->path.DIRECTORY_SEPARATOR);
        $this->maxDatasetSize = 0;
        $rows = 0;
        $steps = 0;
        $inputs = [];
        $tests = [];
        $paths = [];
        foreach($filenames as $filename) {
            $sepfilename = explode(DIRECTORY_SEPARATOR,substr($filename,$prefixLength));
            $label = $sepfilename[0];
            if(count($sepfilename)<2) {
                continue;
            }
            $content = $this->readContents($filename);
            if($this->batchSize==0) {
                // stream mode
                if($this->unclassified) {
                    $data = $content;
                } else {
                    $data = [$content,$label];
                }
                yield $rows => $data;
                $rows++;
                continue;
            }
            $inputs[] = $content;
            $tests[] = $label;
            $paths[] = $filename;
            $rows++;
            if($rows>=$this->batchSize) {
                $inputs = $this->makeBatchInputs($inputs);
                $tests = $this->makeBatchTests($tests);
                $inputsets = [$inputs,$tests];
                if($this->filter) {
                    $inputsets = $this->filter->translate($inputs,$tests,$paths);
                }
                $this->maxDatasetSize += $rows;
                $rows = 0;
                if($this->unclassified) {
                    $data = $inputsets[0];
                } else {
                    $data = $inputsets;
                }
                yield $steps => $data;
                $steps++;
                $this->maxSteps = max($this->maxSteps,$steps);
                $inputs = [];
                $tests = [];
                $paths = [];
            }
        }
        $this->maxDatasetSize += $rows;
        if($this->batchSize==0) {
            // stream mode
            return;
        }
        if($rows) {
            $inputs = $this->makeBatchInputs($inputs);
            $tests = $this->makeBatchTests($tests);
            $inputsets = [$inputs,$tests];
            if($this->filter) {
                $inputsets = $this->filter->translate($inputs,$tests,$paths);
            }
            if($this->unclassified) {
                $data = $inputsets[0];
            } else {
                $data = $inputsets;
            }
            yield $steps => $data;
            $steps++;
            $this->maxSteps = max($this->maxSteps,$steps);
        }
    }
}
